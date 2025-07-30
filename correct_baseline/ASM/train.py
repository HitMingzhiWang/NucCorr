# 修改后的训练流程，适用于 ASM + PointCloud 监督任务

import os
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
from model.ASM import ASMPointPredictor  
from dataset.dataset import NucCorrDataset, RandomShift3DWrapper
from torchvision import transforms
import random


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def chamfer_distance(points1, points2):
    # points1, points2: (B, N, 3)
    # 返回 batch 平均倒角距离
    B, N, _ = points1.shape
    _, M, _ = points2.shape

    # 扩展维度以便广播
    points1_expand = points1.unsqueeze(2)  # (B, N, 1, 3)
    points2_expand = points2.unsqueeze(1)  # (B, 1, M, 3)
    dist = torch.norm(points1_expand - points2_expand, dim=3)  # (B, N, M)

    # 对每个点找最近距离
    min_dist1, _ = torch.min(dist, dim=2)  # (B, N)
    min_dist2, _ = torch.min(dist, dim=1)  # (B, M)

    cd = (min_dist1.mean(dim=1) + min_dist2.mean(dim=1)).mean()  # batch mean
    return cd


def point_loss(pred, target):
    # pred, target: (B, N, 3)
    # pytorch3d chamfer_distance 需要float32, 并且输入shape为(B, N, 3)
    loss= chamfer_distance(pred, target)
    return loss

def train_one_epoch(model, loader, optimizer, device, epoch, scaler, is_main_process):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", disable=not is_main_process)

    for batch in pbar:
        
        seg, img, _, points_gt = batch
        seg, img, points_gt = seg.to(device), img.to(device), points_gt.to(device)
        x = torch.cat([seg, img], dim=1)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            pred_points = model(x)
            loss = point_loss(pred_points, points_gt)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        if is_main_process:
            wandb.log({'train/loss': loss.item(), 'train/epoch': epoch})

    return total_loss / len(loader)

def validate(model, loader, device, epoch, is_main_process):
    model.eval()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", disable=not is_main_process)

    with torch.no_grad():
        for batch in pbar:
            seg, img, _, points_gt = batch
            seg, img, points_gt = seg.to(device), img.to(device), points_gt.to(device)
            x = torch.cat([seg, img], dim=1)
            pred_points = model(x)
            loss = point_loss(pred_points, points_gt)

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    avg_loss = total_loss / len(loader)
    if is_main_process:
        wandb.log({'val/loss': avg_loss, 'val/epoch': epoch})
    return avg_loss

def main(args):
    # ===== Setup =====
    set_seed(args.seed)
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = (rank == 0)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if is_distributed:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(device)

    # ===== Load PCA Model =====
    pca_data = np.load(args.pca_path)
    p_mean = torch.from_numpy(pca_data['mean_shape']).float()
    p_components = torch.from_numpy(pca_data['components']).float()
    model = ASMPointPredictor(p_mean, p_components).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ===== DataLoader =====
    train_transform = transforms.Compose([RandomShift3DWrapper(max_shift=10)])
    train_dataset = NucCorrDataset(args.data_dir, args.train_file, train_transform,points_npz_path='/nvme2/mingzhi/NucCorr/nuclei_points_aligned.npz')
    val_dataset = NucCorrDataset(args.data_dir, args.val_file, None,points_npz_path='/nvme2/mingzhi/NucCorr/nuclei_points_aligned.npz')

    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

    train_loader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=12)
    val_loader = DataLoader(val_dataset, args.batch_size, sampler=val_sampler, shuffle=False, num_workers=12)

    # ===== Optimization =====
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs // 3, T_mult=2, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler()

    # ===== WandB =====
    if is_main_process:
        os.environ['WANDB_MODE'] = 'offline'
        os.environ['WANDB_API_KEY'] = args.wandb_key
        wandb.init(project=args.project_name, config=vars(args))

    # ===== Training Loop =====
    for epoch in range(args.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, scaler, is_main_process)
        val_loss = validate(model, val_loader, device, epoch, is_main_process)
        scheduler.step()

        if is_main_process:
            print(f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            if epoch == 0 or val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.module.state_dict() if is_distributed else model.state_dict(), 'best_model.pth')

    if is_main_process:
        wandb.finish()
    cleanup_distributed()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/nvme2/mingzhi/NucCorr/NucCorrData/normal_nuclei/normal_nuclei')
    parser.add_argument('--train_file', type=str, default='/nvme2/mingzhi/NucCorr/NucCorrData/normal_nuclei/normal_nuclei/normal_nuclei_train_10k.txt')
    parser.add_argument('--val_file', type=str, default='/nvme2/mingzhi/NucCorr/NucCorrData/normal_nuclei/normal_nuclei/normal_nuclei_val.txt')
    parser.add_argument('--pca_path', type=str, default='/nvme2/mingzhi/NucCorr/shape_model.npz')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--wandb_key', type=str, default='c414fdb4c81a8cd049cb5889fc118e1bfae04b66')
    parser.add_argument('--project_name', type=str, default='asm_pointcloud')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    main(args)