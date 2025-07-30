import os
import numpy as np
import torch
from skimage import io
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 添加项目路径
sys.path.append('/nvme2/mingzhi/NucCorr')

# 尝试导入EmbedNet，如果失败则使用占位符
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_3DCNN", "/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/3DCNN/model/3DCNN.py")
    model_3DCNN = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_3DCNN)
    EmbedNet = model_3DCNN.EmbedNetUNet
except ImportError:
    print("Warning: Could not import EmbedNet from 3DCNN, using placeholder")
    class EmbedNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 16
        def forward(self, x):
            # 简单的占位符实现
            return torch.randn(x.shape[0], self.embed_dim, x.shape[2]//8, x.shape[3]//8, x.shape[4]//8)

from correct_baseline.utils.error_helper import *

def extract_point_embeddings_fast(voxel_embeddings, all_coords, D_prime, H_prime, W_prime):
    # 预计算所有邻域偏移
    z_offsets = np.arange(-1, 2)   # 深度方向3层
    y_offsets = np.arange(-3, 4)   # 高度方向7像素
    x_offsets = np.arange(-3, 4)   # 宽度方向7像素
    Z, Y, X = np.meshgrid(z_offsets, y_offsets, x_offsets, indexing='ij')
    offsets = np.stack([Z.ravel(), Y.ravel(), X.ravel()], axis=1)  # [147, 3] 因为3 * 7 * 7=147
    
    # 扩展坐标 [N, 1, 3] + [1, 147, 3] => [N, 147, 3]
    all_positions = all_coords[:, None, :] + offsets[None, :, :]
    
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
    embed_values = embed_values.T.reshape(len(all_coords), len(offsets), -1)
    return embed_values.mean(axis=1).cpu().numpy()

def extract_point_embeddings(voxel_embeddings, feature_coords, D_prime, H_prime, W_prime):
    """提取点云嵌入特征 - 兼容性函数"""
    return extract_point_embeddings_fast(voxel_embeddings, feature_coords, D_prime, H_prime, W_prime)

# 设置设备
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

embednet = EmbedNet().to(device)
ckpt_path = "/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/3DCNN/logs/train_20250728-163226/checkpoint_best.pth"
if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device)
    embednet.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ 加载3D CNN权重: {ckpt_path}")
else:
    print(f"✗ 未找到3D CNN权重: {ckpt_path}")
embednet.eval()

# 设置输出目录
output_dir = "/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/pointNet/dataset/processed_data"
os.makedirs(output_dir, exist_ok=True)

# 线程锁用于打印
print_lock = threading.Lock()

def process_single_file(args):
    """处理单个文件的函数"""
    volume_name, seg_name, idx, total = args
    
    with print_lock:
        print(f"Processing [{idx+1}/{total}]: {os.path.basename(volume_name)}")
    
    try:
        img = io.imread(volume_name)
        seg = io.imread(seg_name)
        
        # 获取文件名（不包含扩展名）
        vol_name = os.path.basename(volume_name).split('.')[0]
        
        # 构建掩码：正样本(12)和负样本(13, 23, 123)
        mask_12 = np.logical_or(seg==1, seg==2)  # 正样本：同一个细胞核
        mask_13 = np.logical_or(seg==1, seg==3)  # 负样本：不同细胞核
        mask_23 = np.logical_or(seg==2, seg==3)  # 负样本：不同细胞核
        mask_123 = np.logical_or(np.logical_or(seg==1, seg==2), seg==3)  # 负样本：多个细胞核
        
        # 随机选择一种负样本类型
        negative_masks = [mask_13, mask_23, mask_123]
        mask_negative = negative_masks[np.random.randint(0, len(negative_masks))]
        
        # 检查是否有足够的正样本和负样本
        if not np.any(mask_12):
            with print_lock:
                print(f"Warning: No positive samples (mask_12) for {vol_name}, skipping...")
            return False
            
        if not np.any(mask_negative):
            with print_lock:
                print(f"Warning: No negative samples for {vol_name}, skipping...")
            return False

        # 采样正样本点云
        positive_coords = sample_points(get_boundary(mask_12), n=2048, all=False)
        if len(positive_coords) == 0:
            with print_lock:
                print(f"Warning: No positive points sampled for {vol_name}, skipping...")
            return False
        
        # 修复坐标形状：从 (1, N, 3) 转换为 (N, 3)
        if positive_coords.ndim == 3 and positive_coords.shape[0] == 1:
            positive_coords = positive_coords[0]  # 去掉第一个维度
            
        # 采样负样本点云
        negative_coords = sample_points(get_boundary(mask_negative), n=2048, all=False)
        if len(negative_coords) == 0:
            with print_lock:
                print(f"Warning: No negative points sampled for {vol_name}, skipping...")
            return False
            
        # 修复坐标形状：从 (1, N, 3) 转换为 (N, 3)
        if negative_coords.ndim == 3 and negative_coords.shape[0] == 1:
            negative_coords = negative_coords[0]  # 去掉第一个维度

        # 构建EmbedNet输入：4通道 [1,4,D,H,W]
        if len(img.shape) == 3:
            input_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
        else:
            input_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)

        # 通过EmbedNet
        with torch.no_grad():
            voxel_embeddings = embednet(input_tensor/ 255.0)

        # 将采样点坐标转换为特征图空间坐标
        D, H, W = img.shape
        D_prime, H_prime, W_prime = voxel_embeddings.shape[2:]
        # 处理正样本
        if positive_coords.shape[1] != 3:
            with print_lock:
                print(f"Warning: positive_coords shape is {positive_coords.shape}, expected (N, 3)")
            return False
            
        positive_feature_coords = (positive_coords * np.array([D_prime/D, H_prime/H, W_prime/W])).astype(int)
        positive_embeddings = extract_point_embeddings(voxel_embeddings, positive_feature_coords, D_prime, H_prime, W_prime)
        
        # 处理负样本
        if negative_coords.shape[1] != 3:
            with print_lock:
                print(f"Warning: negative_coords shape is {negative_coords.shape}, expected (N, 3)")
            return False
            
        negative_feature_coords = (negative_coords * np.array([D_prime/D, H_prime/H, W_prime/W])).astype(int)
        negative_embeddings = extract_point_embeddings(voxel_embeddings, negative_feature_coords, D_prime, H_prime, W_prime)

        # 保存正样本npz
        positive_output_path = os.path.join(output_dir, f"{vol_name}_positive.npz")
        np.savez(
            positive_output_path,
            points=positive_coords,
            embeddings=positive_embeddings,
            labels=1
        )
        
        # 保存负样本npz
        negative_output_path = os.path.join(output_dir, f"{vol_name}_negative.npz")
        np.savez(
            negative_output_path,
            points=negative_coords,
            embeddings=negative_embeddings,
            labels=0
        )
        
        with print_lock:
            print(f"✓ Saved {vol_name}_positive.npz ({len(positive_coords)} points) and {vol_name}_negative.npz ({len(negative_coords)} points)")
        
        return True
        
    except Exception as e:
        with print_lock:
            print(f"✗ Error processing {volume_name}: {str(e)}")
            import traceback
            traceback.print_exc()
        return False


if __name__ == "__main__":
    # 读取文件列表
    with open("/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/data/train.txt", "r") as f:
        lines = f.readlines()
        volume_list = [os.path.join("/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/data/img", line.strip()) for line in lines]
        seg_list = [os.path.join("/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/data/seg", line.strip()) for line in lines]

    print(f"Found {len(volume_list)} files to process")

    # 准备任务参数
    tasks = [(volume_name, seg_name, idx, len(volume_list)) for idx, (volume_name, seg_name) in enumerate(zip(volume_list, seg_list))]

    # 使用多线程处理
    max_workers = min(4, len(tasks))  # 限制最大线程数
    print(f"Using {max_workers} threads for parallel processing")

    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_single_file, task): task for task in tasks}
        
        # 处理完成的任务
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                if result:
                    success_count += 1
            except Exception as e:
                with print_lock:
                    print(f"Task failed: {e}")

    print(f"Processing completed! Successfully processed {success_count}/{len(tasks)} files.")
