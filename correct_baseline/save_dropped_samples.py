import os
import torch
import numpy as np
import tifffile
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from complete_model.dataset.dataset import NucCorrDataset

def save_samples(dataset, output_dir):
    """保存数据集中的样本"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(len(dataset)):
        seg, img, complete_mask = dataset[i]
        
        # 创建样本目录
        sample_dir = os.path.join(output_dir, f'sample_{i+1}')
        os.makedirs(sample_dir, exist_ok=True)
        
        # 保存drop后的分割图和图像
        # 转换为numpy数组并确保正确的数据类型
        seg_np = seg.squeeze().numpy().astype(np.uint8)
        img_np = (img.squeeze().numpy() * 255).astype(np.uint8)  # 转换回0-255范围
        complete_mask_np = complete_mask.squeeze().numpy().astype(np.uint8)
        
        tifffile.imwrite(os.path.join(sample_dir, 'input_seg.tiff'), seg_np)
        tifffile.imwrite(os.path.join(sample_dir, 'input_img.tiff'), img_np)
        tifffile.imwrite(os.path.join(sample_dir, 'complete_mask.tiff'), complete_mask_np)
        
        print(f"Saved sample {i+1}/20")

def main():
    # 设置路径
    dataset_dir = "/nvme2/mingzhi/NucCorr/NucCorrData/normal_nuclei/normal_nuclei"
    output_dir = "dropped_samples"
    
    # 读取验证集文件列表
    val_list_path = os.path.join(dataset_dir, "normal_nuclei_val.txt")
    
    # 创建数据集
    dataset = NucCorrDataset(
        root_dir=dataset_dir,
        split_file=val_list_path,
        transform=None,
        img_size=(128, 128, 128),
        train=True,  # 设置为True以启用drop操作
        seed=42
    )
    
    # 选择前20个样本
    indices = list(range(20))
    print(f"Selected {len(indices)} samples for saving")
    
    # 保存样本
    samples = [dataset[idx] for idx in indices]
    save_samples(samples, output_dir)
    
    print(f"\nSaving completed!")
    print(f"Saved {len(indices)} samples to {output_dir}")

if __name__ == "__main__":
    main() 