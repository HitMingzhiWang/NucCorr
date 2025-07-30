import os
import torch
import numpy as np
import tifffile
from pathlib import Path
import argparse
from model.unet import get_model
import joblib
import sys
import random
from torch.utils.data import DataLoader
sys.path.append('/nvme2/mingzhi/NucCorr')
from NucDet.Zernike import compute_zernike_descriptor_from_tensor
from regular_loss.zernike_loss import DifferentiableGMM
from dataset.dataset import NucCorrDataset

def load_gmm_model(gmm_path):
    """加载GMM模型并转换为PyTorch可微分版本"""
    gmm = joblib.load(gmm_path)
    
    # 提取GMM参数
    n_components = gmm.n_components
    n_features = gmm.means_.shape[1]  # 121维zernike特征
    means = gmm.means_
    covariances = gmm.covariances_  # 对于diag类型，这是方差向量
    weights = gmm.weights_
    
    # 创建可微分GMM
    differentiable_gmm = DifferentiableGMM(
        n_components=n_components,
        n_features=n_features,
        means=means,
        covariances=covariances,
        weights=weights,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    return differentiable_gmm

def gumbel_sigmoid(logits, tau=0.5, hard=False, eps=1e-10):
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + eps) + eps)
    y = torch.sigmoid((logits + g) / tau)
    if hard:
        y_hard = (y > 0.5).float()
        y = (y_hard - y).detach() + y  # Straight-through estimator
    return y

def test_single_sample(seg, img, complete_mask, sample_id, model, gmm_model, device, max_zernike_order, save_dir=None):
    """测试单个样本"""
    # 移动到GPU
    seg = seg.to(device)
    img = img.to(device)

    # 模型预测
    with torch.no_grad():
        x = torch.cat([seg, img], dim=1)
        output = model(x)
        # 获取soft probability map (sigmoid输出)
        soft_pred = torch.sigmoid(output)
        # 获取二值化预测
        pred = (soft_pred > 0.5).float()
        
        # 计算zernike特征
        soft_mask = gumbel_sigmoid(output).squeeze(0).squeeze(0)  # 移除batch和channel维度，得到(D, H, W)
        
        # 清理不需要的变量以释放内存
        del output, x
        torch.cuda.empty_cache()
        
        try:
            # 转换为float64并确保在正确的设备上
            soft_mask_float64 = soft_mask.to(torch.float64)
            
            # 检查数据是否包含有效值
            if torch.all(soft_mask_float64 == 0):
                print(f"样本 {sample_id}: 警告: 所有值都为0，无法计算zernike特征")
                gmm_loss = None
            else:
                zernike_feature = compute_zernike_descriptor_from_tensor(
                    soft_mask_float64, 
                    max_order=max_zernike_order, 
                    device=device
                )
                
                # 计算GMM loss
                zernike_feature_batch = zernike_feature.unsqueeze(0)  # 添加batch维度 (1, 121)
                gmm_loss = gmm_model.negative_log_likelihood(zernike_feature_batch)
                
                print(f"样本 {sample_id}: GMM Loss (NLL): {gmm_loss.item():.6f}")
                
                # 清理zernike相关变量
                del zernike_feature, zernike_feature_batch
                torch.cuda.empty_cache()
                
                # 保存结果
                if save_dir is not None:
                    sample_save_dir = os.path.join(save_dir, f'sample_{sample_id}')
                    os.makedirs(sample_save_dir, exist_ok=True)
                    
                    # 保存input image (转换回0-255范围)
                    input_img_np = (img.squeeze().cpu().numpy() * 255).astype(np.uint8)
                    tifffile.imwrite(os.path.join(sample_save_dir, 'input_img.tiff'), input_img_np)
                    
                    # 保存input mask (dropped segmentation)
                    input_mask_np = seg.squeeze().cpu().numpy().astype(np.uint8)
                    tifffile.imwrite(os.path.join(sample_save_dir, 'input_mask.tiff'), input_mask_np)
                    
                    # 保存complete mask (ground truth)
                    complete_mask_np = complete_mask.squeeze().cpu().numpy().astype(np.uint8)
                    tifffile.imwrite(os.path.join(sample_save_dir, 'complete_mask.tiff'), complete_mask_np)
                    
                    # 保存prediction (soft probability map)
                    pred_soft_np = soft_pred.squeeze().cpu().numpy().astype(np.float32)
                    tifffile.imwrite(os.path.join(sample_save_dir, 'pred_soft.tiff'), pred_soft_np)
                    
                    # 保存prediction (binary)
                    pred_binary_np = pred.squeeze().cpu().numpy().astype(np.uint8)
                    tifffile.imwrite(os.path.join(sample_save_dir, 'pred_binary.tiff'), pred_binary_np)
                    
                    print(f"样本 {sample_id}: 结果已保存到 {sample_save_dir}")
                
                return gmm_loss.item()
            
        except Exception as e:
            print(f"样本 {sample_id}: 计算zernike特征时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # 最终清理
        del seg, img, soft_pred, pred, soft_mask, soft_mask_float64
        torch.cuda.empty_cache()
        
        return None

def test_val_dataset(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model = get_model().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 加载GMM模型
    gmm_model = load_gmm_model(args.gmm_path)
    gmm_model.eval()

    # 创建保存目录
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"结果将保存到: {args.save_dir}")

    # 构建validation dataset
    print("构建validation dataset...")
    val_dataset = NucCorrDataset(
        root_dir=args.data_dir,
        split_file=args.val_file,
        transform=None,  # 测试时不使用transform
        img_size=args.img_size,
        train=True,  # 测试模式，不进行数据增强
        seed=args.seed,
        shift_augment=True
    )
    
    print(f"Validation dataset包含 {len(val_dataset)} 个样本")
    
    # 随机选择样本进行测试
    if args.num_samples > len(val_dataset):
        args.num_samples = len(val_dataset)
        print(f"请求的样本数超过数据集大小，将测试所有 {len(val_dataset)} 个样本")
    
    # 随机选择索引
    selected_indices = random.sample(range(len(val_dataset)), args.num_samples)
    print(f"随机选择了 {len(selected_indices)} 个样本进行测试")
    
    # 测试选中的样本
    gmm_losses = []
    for i, idx in enumerate(selected_indices):
        print(f"\n测试样本 {i+1}/{len(selected_indices)}: 索引 {idx}")
        
        # 从dataset获取样本
        seg, img, complete_mask = val_dataset[idx]
        
        # 添加batch维度
        seg = seg.unsqueeze(0)  # (1, 1, D, H, W)
        img = img.unsqueeze(0)  # (1, 1, D, H, W)
        complete_mask = complete_mask.unsqueeze(0)  # (1, 1, D, H, W)
        
        gmm_loss = test_single_sample(
            seg, img, complete_mask, idx, 
            model, gmm_model, device, 
            args.max_zernike_order, args.save_dir
        )
        if gmm_loss is not None:
            gmm_losses.append(gmm_loss)
    
    # 统计结果
    if gmm_losses:
        gmm_losses = np.array(gmm_losses)
        print(f"\n=== 测试结果统计 ===")
        print(f"成功测试样本数: {len(gmm_losses)}")
        print(f"GMM Loss 均值: {gmm_losses.mean():.6f}")
        print(f"GMM Loss 标准差: {gmm_losses.std():.6f}")
        print(f"GMM Loss 最小值: {gmm_losses.min():.6f}")
        print(f"GMM Loss 最大值: {gmm_losses.max():.6f}")
        print(f"GMM Loss 中位数: {np.median(gmm_losses):.6f}")
        
        # 保存统计结果
        if args.save_dir:
            stats_file = os.path.join(args.save_dir, 'test_statistics.txt')
            with open(stats_file, 'w') as f:
                f.write(f"测试样本数: {len(gmm_losses)}\n")
                f.write(f"GMM Loss 均值: {gmm_losses.mean():.6f}\n")
                f.write(f"GMM Loss 标准差: {gmm_losses.std():.6f}\n")
                f.write(f"GMM Loss 最小值: {gmm_losses.min():.6f}\n")
                f.write(f"GMM Loss 最大值: {gmm_losses.max():.6f}\n")
                f.write(f"GMM Loss 中位数: {np.median(gmm_losses):.6f}\n")
            print(f"统计结果已保存到: {stats_file}")
    else:
        print("没有成功计算任何GMM loss")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/nvme2/mingzhi/NucCorr/NucCorrData/normal_nuclei/normal_nuclei', help='Path to data directory')
    parser.add_argument('--val_file', type=str, default='/nvme2/mingzhi/NucCorr/NucCorrData/normal_nuclei/normal_nuclei/normal_nuclei_val.txt', help='Path to validation split file')
    parser.add_argument('--model_path', type=str, default='/nvme2/mingzhi/NucCorr/complete_model/checkpoints/best_model_epoch_no_GMM.pth', help='Path to trained model')
    parser.add_argument('--gmm_path', type=str, default='/nvme2/mingzhi/NucCorr/NucDet/gmm_zernike_model.pkl', help='Path to GMM model')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to test')
    parser.add_argument('--max_zernike_order', type=int, default=20, help='Maximum order for Zernike feature computation')
    parser.add_argument('--seed', type=int, default=68, help='Random seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default='/nvme2/mingzhi/NucCorr/complete_model/test_results', help='Directory to save test results')
    parser.add_argument('--img_size', type=tuple, default=(128, 128, 128), help='Image size (depth, height, width)')
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    test_val_dataset(args) 