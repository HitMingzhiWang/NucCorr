import os
import torch
import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Optional
import random
from torchvision import transforms
import sys
sys.path.append('/nvme2/mingzhi/NucCorr')
from correct_baseline.utils.error_helper import get_boundary, sample_points

class PointCloudDataset(Dataset):
    def __init__(self, 
                 img_dir: str, 
                 seg_dir: str, 
                 annotation_file: str, 
                 num_points: int = 1024,
                 is_training: bool = True,
                 augment_prob: float = 0.8,
                 normalize: bool = True):
                 
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.num_points = num_points
        self.is_training = is_training
        self.augment_prob = augment_prob
        self.normalize = normalize
        
        # 加载标注
        self.annotations = self._load_annotations(annotation_file)
        
        # 打印数据集统计信息
        self._print_dataset_stats()

    def _load_annotations(self, file_path: str) -> List[Dict]:
        """加载文本标注文件"""
        annotations = []
        with open(file_path, 'r') as f:
            for line in f:
                # 跳过空行和注释行
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # 解析行: id mask_ids label
                parts = line.split()
                if len(parts) < 3:
                    continue
                
                sample_id = parts[0].strip()
                mask_ids = [int(x) for x in parts[1].split(',')]
                label = int(parts[2].strip())
                
                annotations.append({
                    'id': sample_id,
                    'mask_ids': mask_ids,
                    'label': label
                })
        return annotations

    def _print_dataset_stats(self):
        """打印数据集统计信息"""
        num_samples = len(self.annotations)
        num_positives = sum(1 for ann in self.annotations if ann['label'] == 1)
        
        print(f"  总样本数: {num_samples}")
        print(f"  正样本数: {num_positives} ({num_positives/num_samples:.1%})")
        print(f"  负样本数: {num_samples - num_positives} ({(num_samples - num_positives)/num_samples:.1%})")
        print(f"  点云点数: {self.num_points}")
        print(f"  数据增强: {'启用' if self.is_training else '禁用'}")
        
    def __len__(self) -> int:
        return len(self.annotations)
    

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ann = self.annotations[idx]
        
        # 构建文件名
        img_filename = f"{ann['id']}.tiff"
        seg_filename = f"{ann['id']}.tiff"
        
        # 加载图像和分割数据
        img_path = os.path.join(self.img_dir, img_filename)
        seg_path = os.path.join(self.seg_dir, seg_filename)
        
        # 使用tifffile加载图像
        seg_data = tiff.imread(seg_path)  # 形状: [D, H, W]
        
        # 创建包含指定mask ID的布尔掩码
        mask = np.isin(seg_data, ann['mask_ids'])
        
        # 只采样zxy坐标
        boundary = get_boundary(mask)
        points = sample_points(boundary, all=False, n=self.num_points)  # shape: (1, N, 3)
        points = points[0]  # (N, 3)
        points_tensor = torch.tensor(points, dtype=torch.float32)
        
        # 应用数据增强 (仅训练集且按概率)
        if self.is_training and random.random() < self.augment_prob:
            points_tensor = self._apply_augmentations(points_tensor)
        
        # 归一化到单位球
        if self.normalize:
            points_tensor = self._normalize_point_cloud(points_tensor)
        
        # 确保点云维度正确 [num_points, 3]
        if points_tensor.size(0) > self.num_points:
            points_tensor = points_tensor[:self.num_points, :]
        
        label_tensor = torch.tensor(ann['label'], dtype=torch.long)
        
        return points_tensor, label_tensor


    def _apply_augmentations(self, points: torch.Tensor) -> torch.Tensor:
        """
        应用随机数据增强到点云
        
        参数:
            points: 点云张量 [num_points, 4] (x, y, z, intensity)
        
        返回:
            增强后的点云张量
        """
        # 随机旋转 (绕Z轴)
        if random.random() < 0.8:
            angle = random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=torch.float32)
            points = torch.matmul(points, rotation_matrix)
        
        # 随机缩放
        if random.random() < 0.8:
            scale_factor = random.uniform(0.8, 1.2)
            points *= scale_factor
        
        # 随机平移
        if random.random() < 0.5:
            translation = torch.rand(3) * 0.1 - 0.05  # [-0.05, 0.05]
            points += translation
        
        # 随机抖动 (位置)
        if random.random() < 0.5:
            jitter = torch.randn_like(points) * 0.02
            points += jitter
        
        return points

    def _normalize_point_cloud(self, points: torch.Tensor) -> torch.Tensor:
        """
        归一化点云
        
        参数:
            points: 点云张量 [num_points, 4]
        
        返回:
            归一化后的点云张量
        """
        # 归一化坐标到单位球
        centroid = torch.mean(points, dim=0)
        points = points - centroid
        max_dist = torch.max(torch.sqrt(torch.sum(points**2, dim=1)))
        if max_dist > 0:
            points = points / max_dist
        return points
