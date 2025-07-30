import os
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from skimage.measure import regionprops, label
import random

class NeuronConnectivityDataset(Dataset):
    def __init__(self, data_root, img_dir, seg_dir, split_file, transform=None, neg_samples=5, use_memmap=True):
        """
        完整的神经元连接数据集类
        
        参数:
        data_root: 数据根目录
        img_dir: 图像体积目录名
        seg_dir: 分割掩码目录名
        split_file: 划分文件路径 (如 'splits/train.txt')
        transform: 3D数据增强函数
        neg_samples: 每个正样本配对的负样本数
        use_memmap: 是否使用内存映射加载大文件
        """
        self.data_root = data_root
        self.img_dir = os.path.join(data_root, img_dir)
        self.seg_dir = os.path.join(data_root, seg_dir)
        self.transform = transform
        self.neg_samples = neg_samples
        self.use_memmap = use_memmap
        
        # 读取划分文件
        with open(os.path.join(data_root, split_file), 'r') as f:
            self.volume_list = [line.strip() for line in f]
        
        # 预加载每个volume的元信息
        self.volumes_meta = []
        for vol_name in self.volume_list:
            # 加载分割掩码
            seg_path = os.path.join(self.seg_dir, vol_name)
            if use_memmap:
                seg = tifffile.memmap(seg_path)
            else:
                seg = tifffile.imread(seg_path)
            
            # 提取区域属性
            labeled_seg = label(seg > 0)
            regions = regionprops(labeled_seg)
            
            pos_regions = []
            neg_regions = []
            
            for region in regions:
                # 获取区域值 (1,2或3)
                val = seg[region.coords[0][0], region.coords[0][1], region.coords[0][2]]
                
                region_data = {
                    'coords': region.coords,
                    'centroid': region.centroid,
                    'bbox': region.bbox
                }
                
                if val in (1, 2):
                    pos_regions.append(region_data)
                elif val == 3:
                    neg_regions.append(region_data)
            
            self.volumes_meta.append({
                'pos_regions': pos_regions,
                'neg_regions': neg_regions,
                'shape': seg.shape
            })
    
    def __len__(self):
        """每个volume生成多个样本对"""
        return len(self.volume_list) * 20  # 假设每个volume生成20个样本对
    
    def __getitem__(self, idx):
        """获取单个样本对"""
        vol_idx = idx // 20
        vol_name = self.volume_list[vol_idx]
        meta = self.volumes_meta[vol_idx]
        
        # 加载完整图像和分割
        img_path = os.path.join(self.img_dir, vol_name)
        seg_path = os.path.join(self.seg_dir, vol_name)
        
        if self.use_memmap:
            img = tifffile.memmap(img_path).astype(np.float32)
            seg = tifffile.memmap(seg_path)
        else:
            img = tifffile.imread(img_path).astype(np.float32)
            seg = tifffile.imread(seg_path)
        
        # 随机选择一对正样本区域
        region_a, region_b = random.sample(meta['pos_regions'], 2)
        
        # 创建三个binary mask通道
        mask_a = np.zeros(seg.shape, dtype=np.float32)
        mask_a[tuple(region_a['coords'].T)] = 1
        
        mask_b = np.zeros(seg.shape, dtype=np.float32)
        mask_b[tuple(region_b['coords'].T)] = 1
        
        mask_union = mask_a | mask_b
        
        # 构建4通道输入张量 [img, mask_a, mask_b, mask_union]
        input_tensor = torch.stack([
            torch.from_numpy(img),
            torch.from_numpy(mask_a),
            torch.from_numpy(mask_b),
            torch.from_numpy(mask_union)
        ], dim=0)  # [4, D, H, W]
        
        # 应用数据增强
        if self.transform:
            input_tensor = self.transform(input_tensor)
            
        # 正样本标签为1
        return input_tensor, torch.tensor(1.0, dtype=torch.float32)
    
    def get_negative_sample(self, vol_idx, region_a):
        """为给定区域生成负样本"""
        meta = self.volumes_meta[vol_idx]
        vol_name = self.volume_list[vol_idx]
        
        # 加载图像和分割
        img_path = os.path.join(self.img_dir, vol_name)
        seg_path = os.path.join(self.seg_dir, vol_name)
        
        if self.use_memmap:
            img = tifffile.memmap(img_path).astype(np.float32)
            seg = tifffile.memmap(seg_path)
        else:
            img = tifffile.imread(img_path).astype(np.float32)
            seg = tifffile.imread(seg_path)
        
        # 随机选择一个负区域
        region_neg = random.choice(meta['neg_regions'])
        
        # 创建mask
        mask_a = np.zeros(seg.shape, dtype=np.float32)
        mask_a[tuple(region_a['coords'].T)] = 1
        
        mask_neg = np.zeros(seg.shape, dtype=np.float32)
        mask_neg[tuple(region_neg['coords'].T)] = 1
        
        mask_union = mask_a | mask_neg
        
        # 构建4通道输入张量
        input_tensor = torch.stack([
            torch.from_numpy(img),
            torch.from_numpy(mask_a),
            torch.from_numpy(mask_neg),
            torch.from_numpy(mask_union)
        ], dim=0)
        
        if self.transform:
            input_tensor = self.transform(input_tensor)
            
        # 负样本标签为0
        return input_tensor, torch.tensor(0.0, dtype=torch.float32)