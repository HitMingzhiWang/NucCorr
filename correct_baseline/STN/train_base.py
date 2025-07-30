import os
from tkinter import N
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
import wandb
import argparse
from datetime import datetime
import numpy as np
from model.stn import TETRIS_NucleiSegmentation
from dataset.dataset import NucCorrDataset, RandomShift3DWrapper
from torchvision import transforms
import random

def cleanup_distributed():
    """Clean up the distributed environment"""
    if dist.is_initialized():
        dist.destroy_process_group()

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scaler, is_main_process):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]', disable=not is_main_process)
    
    for batch_idx, (seg, img, target, target_sdf) in enumerate(pbar):
        seg, img, target, target_sdf = seg.to(device), img.to(device), target.to(device), target_sdf.to(device)
        x = torch.cat([seg, img], dim=1)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            output = model(x)
            loss = criterion(output, target_sdf.float())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        if is_main_process and batch_idx % 10 == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/epoch': epoch,
                'train/step': epoch * len(loader) + batch_idx
            })
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device, epoch, is_main_process):
    model.eval()
    total_loss = 0
    total_dice = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]', disable=not is_main_process)
    
    with torch.no_grad():
        for batch_idx, (seg, img, target,target_sdf) in enumerate(pbar):
            seg, img, target, target_sdf = seg.to(device), img.to(device), target.to(device), target_sdf.to(device)
            x = torch.cat([seg, img], dim=1)
            
            output = model(x)
            loss = criterion(output, target_sdf.float())
            
            pred = (output > 0).float()
            dice = (2. * (pred * target.float()).sum()) / (pred.sum() + target.sum() + 1e-8)
            
            total_loss += loss.item()
            total_dice += dice.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'dice': f'{dice.item():.6f}'
            })
            
            # 可选：每100个batch可视化一次结果
            if is_main_process and batch_idx % 100 == 0:
                # wandb.log({"val/predictions": [wandb.Image(img[0]), wandb.Image(pred[0]), wandb.Image(target[0])]})
                pass

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    
    if is_main_process:
        wandb.log({
            'val/epoch_loss': avg_loss,
            'val/epoch_dice': avg_dice,
            'val/epoch': epoch
        })

    return avg_loss, avg_dice

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, epoch, score):
        if self.mode == 'max':
            score = score
        else:
            score = -score
            
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            
        return self.early_stop

def main(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # ====== 关键修改：添加GPU环境调试信息 ======
    print(f"[Pre-init] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print(f"[Pre-init] Visible CUDA devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    
    # 分布式训练设置
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    
    if is_distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # 关键修改：直接使用LOCAL_RANK对应的设备
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        
        # 添加分布式初始化前的设备信息
        print(f"[Rank {rank}] Pre-init device: {device}")
        print(f"[Rank {rank}] Pre-init current device: {torch.cuda.current_device()}")
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # 添加分布式初始化后的设备信息
        print(f"[Rank {rank}] Post-init device: {device}")
        print(f"[Rank {rank}] Post-init current device: {torch.cuda.current_device()}")
        print(f"[Rank {rank}] Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Single GPU] Using device: {device}")

    is_main_process = (rank == 0)
    sdf_templete = np.load('/nvme2/mingzhi/NucCorr/correct_baseline/STN/nuclei_avg_sdf_template.npy')
    # 模型初始化
    model = TETRIS_NucleiSegmentation(sdf_template=sdf_templete).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # 数据集和数据加载器
    train_transform = transforms.Compose([
        RandomShift3DWrapper(max_shift=128)
    ])
    
    train_dataset = NucCorrDataset(
        root_dir=args.data_dir,
        split_file=args.train_file,
        transform=train_transform,
        train=True,
        seed=args.seed
    )
    
    val_dataset = NucCorrDataset(
        root_dir=args.data_dir,
        split_file=args.val_file,
        transform=None,
        train=False,
        seed=args.seed
    )
    
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=val_sampler
    )
    
    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()
    
    # 学习率调度器 - 使用余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.epochs // 3,  # 每1/3个epoch重启一次
        T_mult=2,  # 每次重启后周期翻倍
        eta_min=args.lr * 0.01  # 最小学习率为初始学习率的1%
    )
    
    # 早停机制
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        mode='max'  # 监控验证集Dice分数
    )
    
    # WandB初始化（仅在主进程）
    if is_main_process:
        os.environ['WANDB_API_KEY'] = args.wandb_key
        os.environ['WANDB_MODE'] = 'offline'
        wandb.init(
            project=args.project_name,
            name=f"{args.project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    # 训练循环
    best_dice = 0
    for epoch in range(args.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler, is_main_process)
        val_loss, val_dice = validate(model, val_loader, criterion, device, epoch, is_main_process)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        if is_main_process:
            print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val Dice: {val_dice:.6f}, LR: {current_lr:.6f}")
            
            # 记录到wandb
            wandb.log({
                'train/loss': train_loss,
                'val/loss': val_loss,
                'val/dice': val_dice,
                'train/lr': current_lr,
                'epoch': epoch
            })
            
            if val_dice > best_dice:
                best_dice = val_dice
                print(f"New best Dice: {best_dice:.6f}! Saving model...")
                os.makedirs('checkpoints', exist_ok=True)
                model_to_save = model.module if is_distributed else model
                torch.save(model_to_save.state_dict(), f'checkpoints/best_model_epoch_{epoch}.pth')
            
            # 检查是否需要早停
            if early_stopping(epoch, val_dice):
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Best epoch was {early_stopping.best_epoch} with score {early_stopping.best_score:.6f}")
                break
    
    # 清理和完成
    if is_main_process:
        wandb.finish()
    cleanup_distributed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/nvme2/mingzhi/NucCorr/NucCorrData/normal_nuclei/normal_nuclei', help='Path to data directory')
    parser.add_argument('--train_file', type=str, default='/nvme2/mingzhi/NucCorr/NucCorrData/normal_nuclei/normal_nuclei/normal_nuclei_train.txt', help='Path to train split file')
    parser.add_argument('--val_file', type=str, default='/nvme2/mingzhi/NucCorr/NucCorrData/normal_nuclei/normal_nuclei/normal_nuclei_val.txt', help='Path to validation split file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of data loading workers')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--project_name', type=str, default='nuc_corr', help='Project name for wandb')
    parser.add_argument('--wandb_key', type=str, default='c414fdb4c81a8cd049cb5889fc118e1bfae04b66', help='Wandb API key')
    parser.add_argument('--seed', type=int, default=618, help='Random seed for reproducibility')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.0001, help='Minimum change in monitored value to qualify as an improvement')
    args = parser.parse_args()
    
    main(args)