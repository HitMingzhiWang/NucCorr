#!/usr/bin/env python3
"""
PointNet2对比学习训练脚本
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append('/nvme2/mingzhi/NucCorr')

from correct_baseline.contrastive_learning.pointNet.dataset.pointnet_dataset import PointNetContrastiveDataset, create_dataloaders
from correct_baseline.contrastive_learning.pointNet.model.pointnet2 import PointNet2Contrastive



class Trainer:
    """训练器类"""
    
    def __init__(self, model, train_loader, val_loader, device, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.args = args
        
        # 损失函数 - 只使用BCE Loss进行二分类
        self.classification_criterion = nn.BCELoss()
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=args.lr_step, 
            gamma=args.lr_gamma
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # 最佳验证准确率
        self.best_val_acc = 0
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_classification_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # 获取数据
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            classification_output, projection_features = self.model(features, return_features=True)
            
            # 计算损失 - 只使用BCE Loss
            # 将labels转换为float类型，output通过sigmoid激活
            labels_float = labels.float()
            classification_loss = self.classification_criterion(classification_output.squeeze(), labels_float)
            
            # 总损失
            total_loss_value = classification_loss
            
            # 反向传播
            total_loss_value.backward()
            
            # 梯度裁剪
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            
            self.optimizer.step()
            
            # 统计
            total_loss += total_loss_value.item()
            total_classification_loss += classification_loss.item()
            
            # 分类准确率 - 使用sigmoid阈值0.5
            pred = (classification_output.squeeze() > 0.5).float()
            correct += (pred == labels_float).sum().item()
            total += labels.size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{total_loss_value.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        # 计算平均值
        avg_loss = total_loss / len(self.train_loader)
        avg_classification_loss = total_classification_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'classification_loss': avg_classification_loss,
            'accuracy': accuracy
        }
    
    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        total_classification_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # 获取数据
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                classification_output, projection_features = self.model(features, return_features=True)
                
                # 计算损失 - 只使用BCE Loss
                labels_float = labels.float()
                classification_loss = self.classification_criterion(classification_output.squeeze(), labels_float)
                total_loss_value = classification_loss
                
                # 统计
                total_loss += total_loss_value.item()
                total_classification_loss += classification_loss.item()
                
                # 分类准确率 - 使用sigmoid阈值0.5
                pred = (classification_output.squeeze() > 0.5).float()
                correct += (pred == labels_float).sum().item()
                total += labels.size(0)
        
        # 计算平均值
        avg_loss = total_loss / len(self.val_loader)
        avg_classification_loss = total_classification_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'classification_loss': avg_classification_loss,
            'accuracy': accuracy
        }
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'args': self.args,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(self.args.save_dir, 'latest_checkpoint.pth'))
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, os.path.join(self.args.save_dir, 'best_model.pth'))
        
        # 定期保存检查点
        if epoch % self.args.save_interval == 0:
            torch.save(checkpoint, os.path.join(self.args.save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 4))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc')
        plt.plot(self.val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.save_dir, 'training_curves.png'))
        plt.close()
    
    def train(self):
        """完整训练流程"""
        print(f"Starting training for {self.args.epochs} epochs...")
        
        for epoch in range(1, self.args.epochs + 1):
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate_epoch()
            
            # 学习率调度
            self.scheduler.step()
            
            # 记录历史
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.train_accs.append(train_metrics['accuracy'])
            self.val_accs.append(val_metrics['accuracy'])
            
            # 打印结果
            print(f'Epoch {epoch}/{self.args.epochs}:')
            print(f'  Train - Loss: {train_metrics["loss"]:.4f}, Acc: {train_metrics["accuracy"]:.2f}%')
            print(f'  Val   - Loss: {val_metrics["loss"]:.4f}, Acc: {val_metrics["accuracy"]:.2f}%')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # 保存最佳模型
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                print(f'  ✓ New best validation accuracy: {self.best_val_acc:.2f}%')
            
            # 保存检查点
            self.save_checkpoint(epoch, is_best)
            
            # 绘制训练曲线
            if epoch % 10 == 0:
                self.plot_training_curves()
        
        print(f"Training completed! Best validation accuracy: {self.best_val_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='PointNet2对比学习训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, 
                       default='/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/pointNet/dataset/processed_data',
                       help='数据目录')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--max_points', type=int, default=2048, help='最大点数')
    parser.add_argument('--embed_dim', type=int, default=16, help='嵌入特征维度')
    parser.add_argument('--use_embeddings', default=True, action='store_true', help='是否使用嵌入特征')
    
    # 模型参数
    parser.add_argument('--input_dim', type=int, default=3, help='点云坐标维度')
    parser.add_argument('--num_classes', type=int, default=1, help='分类数量（BCE Loss使用1）')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--lr_step', type=int, default=30, help='学习率调度步长')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='学习率调度衰减')
    

    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/pointNet/checkpoints', help='模型保存目录')
    parser.add_argument('--save_interval', type=int, default=10, help='保存间隔')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='auto', help='设备 (auto/cuda/cpu)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 创建数据集和数据加载器
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize_points=True,
        use_embeddings=args.use_embeddings,
        max_points=args.max_points,
        random_rotate=True,
        random_jitter=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 创建模型
    print("Creating model...")
    model = PointNet2Contrastive(
        input_dim=args.input_dim,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,  # 使用1，因为BCE Loss输出单个值
        use_embeddings=args.use_embeddings
    ).to(device)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader, device, args)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main() 