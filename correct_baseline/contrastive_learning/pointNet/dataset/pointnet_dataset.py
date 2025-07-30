import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import glob
from typing import List, Tuple, Optional


class PointNetContrastiveDataset(Dataset):
    """
    PointNet对比学习数据集
    加载npz文件，将点云和嵌入特征concat作为输入特征
    """
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 normalize_points: bool = True,
                 use_embeddings: bool = True,
                 max_points: int = 2048,
                 random_rotate: bool = True,
                 random_jitter: bool = True,
                 jitter_std: float = 0.01,
                 jitter_clip: float = 0.05):
        """
        初始化数据集
        
        Args:
            data_dir: npz文件目录
            split: 数据集分割 ('train', 'val', 'test')
            normalize_points: 是否归一化点云坐标到单位球
            use_embeddings: 是否使用嵌入特征
            max_points: 最大点数
            random_rotate: 是否随机旋转
            random_jitter: 是否添加噪声
            jitter_std: 噪声标准差
            jitter_clip: 噪声裁剪范围
        """
        self.data_dir = data_dir
        self.split = split
        self.normalize_points = normalize_points
        self.use_embeddings = use_embeddings
        self.max_points = max_points
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.jitter_std = jitter_std
        self.jitter_clip = jitter_clip
        
        # 查找所有npz文件
        self.file_paths = self._load_file_paths()
        
        print(f"Loaded {len(self.file_paths)} files from {data_dir}")
        print(f"Split: {split}, Normalize to unit sphere: {normalize_points}, Use embeddings: {use_embeddings}")
    
    def _load_file_paths(self) -> List[str]:
        """加载所有npz文件路径"""
        pattern = os.path.join(self.data_dir, "*.npz")
        file_paths = glob.glob(pattern)
        
        # 按文件名排序
        file_paths.sort()
        
        # 根据split分割数据集
        if self.split == 'train':
            file_paths = file_paths[:int(1 * len(file_paths))]
        if self.split == 'val':
            file_paths = file_paths[int(0.9 * len(file_paths)):int(1 * len(file_paths))]
        return file_paths
    
    def _normalize_points_to_unit_sphere(self, points: np.ndarray) -> np.ndarray:
        """将点云归一化到单位球内"""
        if not self.normalize_points:
            return points
        
        # 计算点云的质心
        centroid = np.mean(points, axis=0)
        
        # 将点云中心化到原点
        points = points - centroid
        
        # 计算到原点的最大距离
        max_distance = np.max(np.linalg.norm(points, axis=1))
        
        # 避免除零
        if max_distance > 1e-8:
            # 归一化到单位球内
            points = points / max_distance
        
        return points
    
    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """归一化点云坐标 - 使用单位球归一化"""
        return self._normalize_points_to_unit_sphere(points)
    
    def _random_rotate_points(self, points: np.ndarray) -> np.ndarray:
        """随机旋转点云"""
        if not self.random_rotate:
            return points
        
        # 生成随机旋转矩阵
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # 绕Z轴旋转
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        # 应用旋转
        points = points @ rotation_matrix.T
        return points
    
    def _add_jitter(self, points: np.ndarray) -> np.ndarray:
        """添加噪声"""
        if not self.random_jitter:
            return points
        
        jitter = np.random.normal(0, self.jitter_std, points.shape)
        jitter = np.clip(jitter, -self.jitter_clip, self.jitter_clip)
        points = points + jitter
        
        return points
    
    def _pad_or_sample_points(self, points: np.ndarray, embeddings: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """填充或采样点云到指定数量"""
        num_points = len(points)
        
        if num_points >= self.max_points:
            # 随机采样
            indices = np.random.choice(num_points, self.max_points, replace=False)
            points = points[indices]
            if embeddings is not None:
                embeddings = embeddings[indices]
        else:
            # 重复填充
            indices = np.random.choice(num_points, self.max_points, replace=True)
            points = points[indices]
            if embeddings is not None:
                embeddings = embeddings[indices]
        
        return points, embeddings
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> dict:
        """获取单个样本"""
        file_path = self.file_paths[idx]
        
        try:
            # 加载数据
            data = np.load(file_path)
            points = data['points'].astype(np.float32)
            embeddings = data['embeddings'].astype(np.float32) if self.use_embeddings else None
            label = data['labels']
            
            # 确保label是标量
            if isinstance(label, np.ndarray):
                label = label.item()
            
            # 归一化点云到单位球
            points = self._normalize_points(points)
            
            # 数据增强
            if self.split == 'train':
                points = self._random_rotate_points(points)
                points = self._add_jitter(points)
            
            # 填充或采样点云
            points, embeddings = self._pad_or_sample_points(points, embeddings)
            
            # 构建特征
            if self.use_embeddings and embeddings is not None:
                # 将点云坐标和嵌入特征concat
                features = np.concatenate([points, embeddings], axis=1)
            else:
                features = points
            
            # 转换为tensor
            features = torch.from_numpy(features).float()
            label = torch.tensor(label, dtype=torch.long)
            
            return {
                'features': features,  # [max_points, 3+embed_dim] 或 [max_points, 3]
                'points': torch.from_numpy(points).float(),  # [max_points, 3]
                'label': label,  # scalar
                'file_path': file_path
            }
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # 返回一个默认样本
            default_features = torch.zeros(self.max_points, 3 + (embeddings.shape[1] if self.use_embeddings else 0))
            return {
                'features': default_features,
                'points': torch.zeros(self.max_points, 3),
                'label': torch.tensor(0),
                'file_path': file_path
            }


def create_dataloaders(data_dir: str, 
                      batch_size: int = 32,
                      num_workers: int = 4,
                      **dataset_kwargs) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    创建数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 工作进程数
        **dataset_kwargs: 传递给Dataset的参数
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader
    
    # 创建数据集
    train_dataset = PointNetContrastiveDataset(data_dir, split='train', **dataset_kwargs)
    val_dataset = PointNetContrastiveDataset(data_dir, split='val', **dataset_kwargs)
    test_dataset = PointNetContrastiveDataset(data_dir, split='test', **dataset_kwargs)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


