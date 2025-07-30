import numpy as np
import torch
from skimage import io
import sys
sys.path.append('/nvme2/mingzhi/NucCorr')
from model.point_net import PointNet2Classification
from correct_baseline.utils.error_helper import *

def normalize_point_cloud(points):
    points = np.squeeze(points,axis=0)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    furthest_distance = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    return points / furthest_distance

def test_ids_same_object(seg_arr, id1, id2, model, device='cuda', n_points=2048):
    """
    seg_arr: 3D numpy array, 分割标签
    id1, id2: 需要判断的两个mask id
    model: 已加载权重的 PointNet2Classification 实例
    device: 'cpu' or 'cuda'
    n_points: 采样点数
    返回: 0（不同物体）或 1（同一物体）
    """
    # 1. 合并mask
    mask = ((seg_arr == id1) | (seg_arr == id2)).astype(np.uint8)
    if np.sum(mask) == 0:
        raise ValueError("两个id在分割中都不存在！")
    # 2. 得到点云（假设你有 sample_points_from_surface 或 sample_points）
    # 这里以 sample_points_from_surface 为例
    points = sample_points(get_boundary(mask), n=n_points,all=False)
    points = normalize_point_cloud(points)
    # 如果点数不足n_points，重复采样补足
    if points.shape[1] < n_points:
        idx = np.random.choice(points.shape[0], n_points, replace=True)
        points = points[idx]
    
    # 3. 推理
    model.to(device)
    model.eval()
    with torch.no_grad():
        points_tensor = torch.from_numpy(points).unsqueeze(0).float().to(device)  # (1, n_points, 3)
        output = model(points_tensor)
        pred = (output > 0.5).int().item()
    return pred

""" model = PointNet2Classification(num_classes=1)
checkpoint_path = "/nvme2/mingzhi/NucCorr/correct_baseline/MIDL/best_pointnet.pth"  # 修改为你的权重文件
model.load_state_dict(torch.load(checkpoint_path, map_location='cuda'))
obj = io.imread('/nvme2/mingzhi/NucCorr/NucCorrData/split_error/match_seg/6653766.tiff')
print(test_ids_same_object(obj,21,1,model)) """
