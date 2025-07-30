import os
import torch
import numpy as np
import tifffile
import argparse
from model.ASM import ASMPointPredictor
import sys
sys.path.append('/nvme2/mingzhi/NucCorr')
from scipy.spatial import ConvexHull, Delaunay

def points_to_convex_hull_mask(points, shape):
    hull = ConvexHull(points)
    delaunay = Delaunay(points[hull.vertices])
    # 生成所有体素坐标
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij'
    )
    coords = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=-1)
    mask = delaunay.find_simplex(coords) >= 0
    return mask.reshape(shape)


def center_crop_3d(img, seg, target_size=(128, 128, 128)):
    """
    从seg的中心点出发，crop img和seg为target_size
    Args:
        img: (D, H, W) 原始图像
        seg: (D, H, W) 原始分割
        target_size: (D, H, W) 目标尺寸
    Returns:
        img_crop: (D, H, W) 裁剪后的图像
        seg_crop: (D, H, W) 裁剪后的分割
    """
    seg_coords = np.nonzero(seg)
    if len(seg_coords[0]) == 0:
        center_d = img.shape[0] // 2
        center_h = img.shape[1] // 2
        center_w = img.shape[2] // 2
    else:
        center_d = int(np.mean(seg_coords[0]))
        center_h = int(np.mean(seg_coords[1]))
        center_w = int(np.mean(seg_coords[2]))
    

    half_d, half_h, half_w = target_size[0] // 2, target_size[1] // 2, target_size[2] // 2
    
    start_d = max(0, center_d - half_d)
    start_h = max(0, center_h - half_h)
    start_w = max(0, center_w - half_w)
    
    end_d = min(img.shape[0], start_d + target_size[0])
    end_h = min(img.shape[1], start_h + target_size[1])
    end_w = min(img.shape[2], start_w + target_size[2])
    
  
    if end_d - start_d < target_size[0]:
        if end_d == img.shape[0]:
            start_d = max(0, end_d - target_size[0])
        else:
            end_d = min(img.shape[0], start_d + target_size[0])
    
    if end_h - start_h < target_size[1]:
        if end_h == img.shape[1]:
            start_h = max(0, end_h - target_size[1])
        else:
            end_h = min(img.shape[1], start_h + target_size[1])
    
    if end_w - start_w < target_size[2]:
        if end_w == img.shape[2]:
            start_w = max(0, end_w - target_size[2])
        else:
            end_w = min(img.shape[2], start_w + target_size[2])
    
 
    img_crop = img[start_d:end_d, start_h:end_h, start_w:end_w]
    seg_crop = seg[start_d:end_d, start_h:end_h, start_w:end_w]
    

    if img_crop.shape != target_size:
        padded_img = np.zeros(target_size, dtype=img_crop.dtype)
        padded_seg = np.zeros(target_size, dtype=seg_crop.dtype)
        
        padded_img[:img_crop.shape[0], :img_crop.shape[1], :img_crop.shape[2]] = img_crop
        padded_seg[:seg_crop.shape[0], :seg_crop.shape[1], :seg_crop.shape[2]] = seg_crop
        
        img_crop = padded_img
        seg_crop = padded_seg
    
    return img_crop, seg_crop

def test_single_sample(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    pca_data = np.load(args.pca_path)
    p_mean = torch.from_numpy(pca_data['mean_shape']).float()
    p_components = torch.from_numpy(pca_data['components']).float()
    model = ASMPointPredictor(p_mean, p_components).to(device)
    

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    

    img = tifffile.imread(args.img_path)
    seg = tifffile.imread(args.seg_path)
    
    print(f"原始图像尺寸: {img.shape}")
    print(f"原始分割尺寸: {seg.shape}")
    

    seg = (seg > 0).astype(np.uint8)
    
  
    img_crop, seg_crop = center_crop_3d(img, seg, target_size=(128, 128, 128))
    
    print(f"裁剪后图像尺寸: {img_crop.shape}")
    print(f"裁剪后分割尺寸: {seg_crop.shape}")
    
  
    img_tensor = torch.from_numpy(img_crop).float().unsqueeze(0).unsqueeze(0) / 255.0  # (1, 1, 128, 128, 128)
    seg_tensor = torch.from_numpy(seg_crop).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 128, 128)
    

    img_tensor = img_tensor.to(device)
    seg_tensor = seg_tensor.to(device)
    
    with torch.no_grad():
        x = torch.cat([seg_tensor, img_tensor], dim=1)  # (1, 2, 128, 128, 128)
        pred_points = model(x)
        
        print(f"预测点云形状: {pred_points.shape}")
    
    # 保存结果
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 保存输入数据
    tifffile.imwrite(os.path.join(args.save_dir, 'input_img.tiff'), img_crop.astype(np.uint8))
    tifffile.imwrite(os.path.join(args.save_dir, 'input_seg.tiff'), seg_crop.astype(np.uint8))
    
    # 保存预测结果
    pred_points_np = pred_points.squeeze(0).cpu().numpy()  # (N, 3)
    np.save(os.path.join(args.save_dir, 'pred_points.npy'), pred_points_np)
    tifffile.imwrite(os.path.join(args.save_dir, 'pred_seg.tiff'), points_to_convex_hull_mask(pred_points_np,[128,128,128]).astype(np.uint8))
    
    print(f"结果已保存到: {args.save_dir}")
    print(f"预测点云数量: {pred_points_np.shape[0]}")
    print(f"点云坐标范围: X[{pred_points_np[:, 0].min():.3f}, {pred_points_np[:, 0].max():.3f}], "
          f"Y[{pred_points_np[:, 1].min():.3f}, {pred_points_np[:, 1].max():.3f}], "
          f"Z[{pred_points_np[:, 2].min():.3f}, {pred_points_np[:, 2].max():.3f}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='/nvme2/mingzhi/NucCorr/NucCorrData/split_error/img/532054_img.tiff', help='Path to input image file')
    parser.add_argument('--seg_path', type=str, default='/nvme2/mingzhi/NucCorr/NucCorrData/split_error/seg/532054.tiff', help='Path to input segmentation file')
    parser.add_argument('--model_path', type=str,  default='/nvme2/mingzhi/NucCorr/correct_baseline/ASM/best_model.pth', help='Path to trained model')
    parser.add_argument('--pca_path', type=str, default='/nvme2/mingzhi/NucCorr/shape_model.npz', help='Path to PCA model')
    parser.add_argument('--save_dir', type=str, default='/nvme2/mingzhi/NucCorr/correct_baseline/ASM/test_results', help='Directory to save results')
    args = parser.parse_args()
    
    test_single_sample(args) 