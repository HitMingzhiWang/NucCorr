import os
import re
import numpy as np
import tifffile
from tqdm import tqdm
from scipy.ndimage import center_of_mass, shift
import torch
from Zernike import compute_zernike_descriptor_from_tensor

torch.set_default_device('cuda')

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return match.group(0) if match else filename


def process_one_file_centered(filename, folder_path, max_order=20, target_size=128, device='cuda'):
    file_path = os.path.join(folder_path, filename)
    mask = tifffile.imread(file_path)
    mask = np.where(mask > 0, 1, 0).astype(np.float32)
    if np.all(mask == 0):
        key = extract_number(os.path.splitext(filename)[0])
        return (None, key, 'all zero')
    # 转为torch tensor并放到device
    mask_tensor = torch.from_numpy(mask).to(torch.float64).to(device)
    zernike_feature = compute_zernike_descriptor_from_tensor(mask_tensor, max_order=max_order, device=device)
    zernike_feature = zernike_feature.detach().cpu().numpy()
    key = extract_number(os.path.splitext(filename)[0])
    return (zernike_feature, key, None)
    

if __name__ == "__main__":
    folder = "/nvme2/mingzhi/NucCorr/NucCorrData/normal_nuclei/normal_nuclei/seg"
    max_order = 20
    target_size = 128
    device = 'cuda'  # 或 'cpu'
    tiff_files = [f for f in os.listdir(folder) if f.lower().endswith(('.tif', '.tiff'))]
    print(f"开始处理 {len(tiff_files)} 个TIFF文件...")
    
    # 使用字典来存储特征，每个ID对应一个特征
    features_dict = {}
    
    for filename in tqdm(tiff_files, desc="Zernike质心对齐", unit="file"):
        features, key, err = process_one_file_centered(filename, folder, max_order, target_size, device)
        if features is not None:
            features_dict[key] = features
    
    if features_dict:
        # 保存为字典结构
        np.savez("features_split_centered.npz", **features_dict)
        print(f"已保存特征到 features_split_centered.npz, 数量: {len(features_dict)}")
        print(f"前5个ID: {list(features_dict.keys())[:5]}")
        print(f"特征维度: {list(features_dict.values())[0].shape}")