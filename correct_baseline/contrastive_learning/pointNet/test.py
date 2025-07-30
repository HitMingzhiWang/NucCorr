#!/usr/bin/env python3
"""
使用3D CNN和PointNet2判断两个mask是否属于同一个物体
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from skimage import io
import cc3d
import importlib.util
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tifffile

# 添加项目路径
sys.path.append('/nvme2/mingzhi/NucCorr')
from correct_baseline.utils.error_helper import *

def load_3d_cnn_model():
    """加载3D CNN模型"""
    try:
        # 动态加载3D CNN模型
        spec = importlib.util.spec_from_file_location(
            "model_3DCNN", 
            "/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/3DCNN/model/3DCNN.py"
        )
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        
        # 创建模型实例
        model = model_module.EmbedNetUNet()
        
        # 加载预训练权重（如果有的话）
        checkpoint_path = "/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/3DCNN/checkpoints/best_model.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ 加载3D CNN预训练权重: {checkpoint_path}")
        
        model.eval()
        return model
        
    except Exception as e:
        print(f"✗ 加载3D CNN模型失败: {e}")
        return None

def load_pointnet2_model():
    """加载PointNet2模型"""
    try:
        from correct_baseline.contrastive_learning.pointNet.model.pointnet2 import PointNet2Contrastive
        
        # 创建模型
        model = PointNet2Contrastive(
            input_dim=3,
            embed_dim=16,
            num_classes=1,
            use_embeddings=True
        )
        
        # 加载预训练权重（如果有的话）
        checkpoint_path = "/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/pointNet/checkpoints/best_model.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ 加载PointNet2预训练权重: {checkpoint_path}")
        
        model.eval()
        return model
        
    except Exception as e:
        print(f"✗ 加载PointNet2模型失败: {e}")
        return None

def extract_center_region(volume, seg, mask_id1, mask_id2, crop_size=128):
    """从两个mask ID的中点crop 128×128×128的区域"""
    # 创建两个mask
    mask1 = (seg == mask_id1).astype(np.uint8)
    mask2 = (seg == mask_id2).astype(np.uint8)
    
    # 找到两个mask的坐标
    coords1 = np.where(mask1 > 0)
    coords2 = np.where(mask2 > 0)
    
    if len(coords1[0]) == 0 or len(coords2[0]) == 0:
        return None, None, None
    
    # 计算两个mask的中心点
    center1 = np.array([
        np.mean(coords1[0]),
        np.mean(coords1[1]), 
        np.mean(coords1[2])
    ])
    center2 = np.array([
        np.mean(coords2[0]),
        np.mean(coords2[1]),
        np.mean(coords2[2])
    ])
    
    # 计算两个中心点的中点
    center_point = (center1 + center2) / 2
    
    # 计算crop区域的边界
    half_size = crop_size // 2
    min_x = int(center_point[0] - half_size)
    max_x = int(center_point[0] + half_size)
    min_y = int(center_point[1] - half_size)
    max_y = int(center_point[1] + half_size)
    min_z = int(center_point[2] - half_size)
    max_z = int(center_point[2] + half_size)
    
    # 确保边界在volume范围内
    min_x = max(0, min_x)
    max_x = min(volume.shape[0], max_x)
    min_y = max(0, min_y)
    max_y = min(volume.shape[1], max_y)
    min_z = max(0, min_z)
    max_z = min(volume.shape[2], max_z)
    
    # 如果crop区域超出边界，调整到边界内
    if max_x - min_x < crop_size:
        if min_x == 0:
            max_x = min(crop_size, volume.shape[0])
        else:
            min_x = max(0, volume.shape[0] - crop_size)
    
    if max_y - min_y < crop_size:
        if min_y == 0:
            max_y = min(crop_size, volume.shape[1])
        else:
            min_y = max(0, volume.shape[1] - crop_size)
    
    if max_z - min_z < crop_size:
        if min_z == 0:
            max_z = min(crop_size, volume.shape[2])
        else:
            min_z = max(0, volume.shape[2] - crop_size)
    
    # 提取crop区域
    crop_volume = volume[min_x:max_x, min_y:max_y, min_z:max_z]
    crop_seg = seg[min_x:max_x, min_y:max_y, min_z:max_z]
    
    if crop_volume.shape != (crop_size, crop_size, crop_size):
        final_volume = np.zeros((crop_size, crop_size, crop_size), dtype=crop_volume.dtype)
        final_seg = np.zeros((crop_size, crop_size, crop_size), dtype=crop_seg.dtype)
        
        # 复制数据到中心位置
        start_x = (crop_size - crop_volume.shape[0]) // 2
        start_y = (crop_size - crop_volume.shape[1]) // 2
        start_z = (crop_size - crop_volume.shape[2]) // 2
        
        end_x = start_x + crop_volume.shape[0]
        end_y = start_y + crop_volume.shape[1]
        end_z = start_z + crop_volume.shape[2]
        
        final_volume[start_x:end_x, start_y:end_y, start_z:end_z] = crop_volume
        final_seg[start_x:end_x, start_y:end_y, start_z:end_z] = crop_seg
        
        crop_volume = final_volume
        crop_seg = final_seg
    
    return crop_volume, crop_seg, (min_x, min_y, min_z)

def extract_embeddings(volume_region, cnn_model, device='cpu'):
    """使用3D CNN提取嵌入特征"""
    try:
        volume_norm = volume_region
        
        # 转换为tensor
        volume_tensor = torch.from_numpy(volume_norm).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, D]
        
        # 移动到设备
        volume_tensor = volume_tensor.to(device)
        cnn_model = cnn_model.to(device)
        
        # 提取特征
        with torch.no_grad():
            embeddings = cnn_model(volume_tensor /255.0)
        
        return embeddings
        
    except Exception as e:
        print(f"✗ 提取嵌入特征失败: {e}")
        return None



def map_features_to_points_fast(voxel_embeddings, points, crop_offset, crop_size=128):
    """使用邻域平均方法将CNN特征映射到点云上"""
    try:
        embed_dim = voxel_embeddings.shape[1]
        D_prime, H_prime, W_prime = voxel_embeddings.shape[2:]
        
        relative_points = points.astype(int)
        
        z_offsets = np.arange(-1, 2)   # 深度方向3层
        y_offsets = np.arange(-3, 4)   # 高度方向7像素
        x_offsets = np.arange(-3, 4)   # 宽度方向7像素
        Z, Y, X = np.meshgrid(z_offsets, y_offsets, x_offsets, indexing='ij')
        offsets = np.stack([Z.ravel(), Y.ravel(), X.ravel()], axis=1)  # [147, 3]
        
        # 扩展坐标 [N, 1, 3] + [1, 147, 3] => [N, 147, 3]
        all_positions = relative_points[:, None, :] + offsets[None, :, :]
        
        # 边界裁剪
        all_positions = np.clip(all_positions, 
                               [0, 0, 0], 
                               [D_prime-1, H_prime-1, W_prime-1])
        
        # 提取嵌入 [N, 147, embed_dim]
        z_indices = all_positions[:, :, 0].astype(int).ravel()
        y_indices = all_positions[:, :, 1].astype(int).ravel()
        x_indices = all_positions[:, :, 2].astype(int).ravel()
        
        # 批量提取特征
        embed_values = voxel_embeddings[0, :, z_indices, y_indices, x_indices]
        
        # 重新组织并计算平均值 [N, embed_dim]
        embed_values = embed_values.T.reshape(len(relative_points), len(offsets), -1)
        mapped_features = embed_values.mean(axis=1)
        
        return mapped_features.cpu().numpy()
        
    except Exception as e:
        print(f"✗ 特征映射失败: {e}")
        return None


def predict_same_object(volume_path, seg_path, mask_id1, mask_id2, 
                       cnn_model, pointnet_model, device='cpu'):
    """预测两个mask是否属于同一个物体（合并mask后一次性输入PointNet2）"""
    
    print(f"正在分析mask {mask_id1} 和 {mask_id2}...")
    
    # 加载数据
    try:
        volume = io.imread(volume_path)
        seg = io.imread(seg_path)
        print(f"✓ 加载数据成功: volume {volume.shape}, seg {seg.shape}")
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        return None
    
    crop_volume, crop_seg, crop_offset = extract_center_region(volume, seg, mask_id1, mask_id2)
    
    if crop_volume is None:
        print("✗ 无法提取crop区域")
        return None
    
    print(f"✓ 提取crop区域成功: {crop_volume.shape}")
    
    embeddings = extract_embeddings(crop_volume, cnn_model, device)
    
    if embeddings is None:
        print("✗ 提取嵌入特征失败")
        return None
    
    print(f"✓ 提取嵌入特征成功: {embeddings.shape}")
    
    merged_mask = ((crop_seg == mask_id1) | (crop_seg == mask_id2)).astype(np.uint8)
    points = sample_points(get_boundary(merged_mask), n=2048, all=False)  # 采样合并区域的点
    points = np.squeeze(points,axis=0)
    if points is None:
        print("✗ 采样点云失败")
        return None
    
    print(f"✓ 采样点云成功: points {points.shape}")

    def normalize_points(points):
        centroid = np.mean(points, axis=0)
        points = points - centroid
        m = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points = points / m
        return points
    
    mapped_features = map_features_to_points_fast(embeddings, points, crop_offset)
    points_norm = normalize_points(points)
    features = np.concatenate([points_norm, mapped_features], axis=1)
    
    features_tensor = torch.from_numpy(features).float().unsqueeze(0)  # [1, N, 3+embed_dim]
    features_tensor = features_tensor.to(device)
    pointnet_model = pointnet_model.to(device)
    

    try:
        with torch.no_grad():
            output, _ = pointnet_model(features_tensor, return_features=True)
        score = output.squeeze().cpu().numpy()
        is_same_object = score > 0.5
        return {
            'is_same_object': bool(is_same_object),
            'confidence_score': float(score),
            'crop_shape': crop_volume.shape,
            'crop_offset': crop_offset
        }
    except Exception as e:
        print(f"✗ PointNet2预测失败: {e}")
        return None


def test_ids_same_object(volume, seg, mask_id1, mask_id2, 
                       cnn_model, pointnet_model, device='cpu'):
    print(f"正在分析mask {mask_id1} 和 {mask_id2}...")   
    crop_volume, crop_seg, crop_offset = extract_center_region(volume, seg, mask_id1, mask_id2)
    
    if crop_volume is None:
        print("✗ 无法提取crop区域")
        return None
    
    print(f"✓ 提取crop区域成功: {crop_volume.shape}")
    
    embeddings = extract_embeddings(crop_volume, cnn_model, device)
    
    if embeddings is None:
        print("✗ 提取嵌入特征失败")
        return None
    
    print(f"✓ 提取嵌入特征成功: {embeddings.shape}")
    
    merged_mask = ((crop_seg == mask_id1) | (crop_seg == mask_id2)).astype(np.uint8)
    points = sample_points(get_boundary(merged_mask), n=2048, all=False)  # 采样合并区域的点
    points = np.squeeze(points,axis=0)
    if points is None:
        print("✗ 采样点云失败")
        return None
    
    print(f"✓ 采样点云成功: points {points.shape}")

    def normalize_points(points):
        centroid = np.mean(points, axis=0)
        points = points - centroid
        m = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points = points / m
        return points
    
    mapped_features = map_features_to_points_fast(embeddings, points, crop_offset)
    points_norm = normalize_points(points)
    features = np.concatenate([points_norm, mapped_features], axis=1)
    
    features_tensor = torch.from_numpy(features).float().unsqueeze(0)  # [1, N, 3+embed_dim]
    features_tensor = features_tensor.to(device)
    pointnet_model = pointnet_model.to(device)
    

    try:
        with torch.no_grad():
            output, _ = pointnet_model(features_tensor, return_features=True)
        score = output.squeeze().cpu().numpy()
        is_same_object = score > 0.5
        return {
            'is_same_object': bool(is_same_object),
            'confidence_score': float(score),
            'crop_shape': crop_volume.shape,
            'crop_offset': crop_offset
        }
    except Exception as e:
        print(f"✗ PointNet2预测失败: {e}")
        return None














def main():

    parser = argparse.ArgumentParser(description='判断两个mask是否属于同一个物体')
    parser.add_argument('--volume', type=str, default='/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/data/img/115107.tiff', help='Volume文件路径')
    parser.add_argument('--seg', type=str, default='/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/data/seg/115107.tiff',help='Segmentation文件路径')
    parser.add_argument('--mask_id1', type=int, default=3, help='第一个mask ID')
    parser.add_argument('--mask_id2', type=int, default=2, help='第二个mask ID')
    parser.add_argument('--device', type=str,default='cuda',  help='设备 (cpu/cuda)')
    parser.add_argument('--output', type=str, default=None, help='输出结果文件路径')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.volume):
        print(f"✗ Volume文件不存在: {args.volume}")
        return
    
    if not os.path.exists(args.seg):
        print(f"✗ Segmentation文件不存在: {args.seg}")
        return
    
    print("=== 加载模型 ===")
    
    # 加载模型
    cnn_model = load_3d_cnn_model()
    pointnet_model = load_pointnet2_model()
    
    if cnn_model is None or pointnet_model is None:
        print("✗ 模型加载失败")
        return
    
    print("✓ 所有模型加载成功")
    
    print("\n=== 开始预测 ===")
    
    # 进行预测
    result = predict_same_object(
        args.volume, 
        args.seg, 
        args.mask_id1, 
        args.mask_id2,
        cnn_model,
        pointnet_model,
        args.device
    )
    
    if result is None:
        print("✗ 预测失败")
        return
    
    # 输出结果
    print("\n=== 预测结果 ===")
    print(f"Mask {args.mask_id1} 和 Mask {args.mask_id2}")
    print(f"是否属于同一个物体: {'是' if result['is_same_object'] else '否'}")
    print(f"置信度分数: {result['confidence_score']:.4f}")

if __name__ == "__main__":
    main() 