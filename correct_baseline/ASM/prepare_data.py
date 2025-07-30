import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_fill_holes, distance_transform_edt
import tifffile
import glob
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from scipy.spatial import procrustes
def fibonacci_sphere(n_samples=96, anisotropy=(1, 1, 1)):
    """
    斐波那契球面采样（考虑各向异性因子）
    参考论文中的公式：
        z_k = -1 + (2*k)/(n-1)
        y_k = sqrt(1 - z_k^2) * sin[2π(1-φ^{-1})k]
        x_k = sqrt(1 - z_k^2) * cos[2π(1-φ^{-1})k]
    
    n_samples: 采样点数量（论文中使用96）
    anisotropy: 各向异性因子 (sx, sy, sz)，例如 (1, 1, 7.1)
    """
    # 黄金比例 φ = (1+√5)/2 ≈ 1.618034
    phi = (1 + np.sqrt(5)) / 2
    
    k = np.arange(n_samples, dtype=np.float32)
    z = -1 + 2 * k / (n_samples - 1)  # z ∈ [-1, 1]
    
    # 计算角度增量 (1-1/φ) ≈ 0.382
    angle = 2 * np.pi * (1 - 1/phi) * k
    
    # 计算xy平面上的半径
    r_xy = np.sqrt(1 - z**2)
    
    # 计算笛卡尔坐标
    x = r_xy * np.cos(angle)
    y = r_xy * np.sin(angle)
    
    # 应用各向异性因子
    sx, sy, sz = anisotropy
    points = np.stack([x * sx, y * sy, z * sz], axis=1)
    
    # 归一化为单位向量
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def find_centroid(mask):
    """计算掩码的质心（距离变换最深点）"""
    distance = distance_transform_edt(mask)
    centroid_idx = np.unravel_index(np.argmax(distance), mask.shape)
    return np.array(centroid_idx, dtype=np.float32)

def get_surface_points(mask):
    """获取所有位于对象表面的点"""
    # 创建内核以检测表面点
    kernel = np.ones((3, 3, 3), dtype=bool)
    kernel[1, 1, 1] = False  # 中心点
    
    # 膨胀掩码以找到边界
    from scipy.ndimage import binary_dilation
    dilated = binary_dilation(mask, structure=kernel)
    
    # 表面点是掩码内但在膨胀后边界上的点
    surface_mask = mask & (dilated ^ mask)
    return np.argwhere(surface_mask)

def ray_cast(origin, direction, mask, max_steps=100, step_size=1.0):
    """
    从内部点沿射线方向投射，找到表面交点
    """
    current_pos = origin.copy()
    inside = mask[tuple(np.floor(current_pos).astype(int))]
    
    # 沿着射线前进直到离开对象
    for _ in range(max_steps):
        # 更新位置
        current_pos += direction * step_size
        
        # 检查是否仍在边界内
        if not all(0 <= p < s for p, s in zip(current_pos, mask.shape)):
            break
        
        # 获取当前网格位置
        grid_pos = tuple(np.floor(current_pos).astype(int))
        
        # 检查是否离开对象
        current_inside = mask[grid_pos]
        if inside and not current_inside:
            # 找到表面点 - 回退一步
            surface_point = current_pos - direction * step_size
            return surface_point
    
    # 如果未找到交点，返回None
    return None

def parameterized_sampling(mask, ray_directions):
    """
    使用球面模板进行参数化表面采样（星形凸起假设）
    mask: 3D二值分割掩码 (H, W, D)
    ray_directions: 预定义的射线方向 (n_points, 3)
    """
    # 1. 预处理掩码
    mask = binary_fill_holes(mask)
    
    # 2. 计算质心（确保在对象内部）
    centroid = find_centroid(mask)
    
    # 3. 获取所有表面点坐标
    surface_points = get_surface_points(mask)
    
    # 4. 计算方向向量并归一化
    dir_vectors = surface_points - centroid
    norms = np.linalg.norm(dir_vectors, axis=1)
    valid = norms > 1e-6  # 避免除以零
    dir_vectors[valid] /= norms[valid][:, np.newaxis]
    
    # 5. 为方向向量创建KD树
    tree = cKDTree(dir_vectors)
    
    # 6. 为每个射线方向找到最近的表面点
    sampled_points = []
    for direction in ray_directions:
        # 找到最接近的方向
        _, idx = tree.query(direction, k=1)
        
        # 使用射线投射找到精确的表面点
        surface_point = ray_cast(centroid, direction, mask)
        
        if surface_point is not None:
            sampled_points.append(surface_point)
        else:
            # 回退到最近邻
            sampled_points.append(surface_points[idx])
    
    return np.array(sampled_points)

def process_single_mask(args):
    """处理单个mask的函数，用于多线程"""
    mask_path, obj_id, ray_directions = args
    try:
        mask = tifffile.imread(mask_path)
        pc = parameterized_sampling(mask, ray_directions)
        return obj_id, pc
    except Exception as e:
        print(f"Error processing {mask_path}: {e}")
        return obj_id, None

def generalized_procrustes(points_list, max_iter=10, tol=1e-5):
    """
    广义Procrustes分析 - 迭代对齐所有形状
    points_list: 对应点云列表 [N, n_points, 3]
    """
    # 初始化平均形状
    mean_shape = np.mean(points_list, axis=0)
    
    # 归一化尺度
    def normalize_scale(points):
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        scale = np.linalg.norm(centered)
        return centered / scale, centroid, scale
    
    aligned_shapes = []
    scales = []
    centroids = []
    
    # 首先归一化所有形状
    for points in points_list:
        scaled, centroid, scale = normalize_scale(points)
        aligned_shapes.append(scaled)
        centroids.append(centroid)
        scales.append(scale)
    
    for _ in range(max_iter):
        # 更新平均形状
        new_mean = np.mean(aligned_shapes, axis=0)
        
        # 归一化平均形状
        new_mean, _, _ = normalize_scale(new_mean)
        
        # 检查收敛
        if np.linalg.norm(new_mean - mean_shape) < tol:
            break
        
        mean_shape = new_mean
        
        # 将每个形状对齐到新的平均形状
        aligned_shapes = []
        for points in points_list:
            # 对齐到当前平均形状
            _, aligned, _ = procrustes(mean_shape, points)
            aligned_shapes.append(aligned)
    
    # 重建最终形状（包括尺度和位置）
    final_shapes = []
    for aligned, centroid, scale in zip(aligned_shapes, centroids, scales):
        # 恢复尺度和位置
        reconstructed = aligned * scale + centroid
        final_shapes.append(reconstructed)
    
    # 计算最终平均形状
    final_mean = np.mean(final_shapes, axis=0)
    
    return final_shapes, final_mean

# ===== 主程序 =====
if __name__ == "__main__":
    # 配置参数（根据论文）
    MASK_DIR = "/nvme2/mingzhi/NucCorr/NucCorrData/normal_nuclei/normal_nuclei/seg"
    N_RAYS = 96  # 论文中使用96条射线
    ANISOTROPY = (1, 1, 1)  # 各向异性因子（针对Parhyale数据集）
    OUT_FILE = "nuclei_points_aligned.npz"
    SHAPE_MODEL_FILE = "shape_model.npz"
    MAX_WORKERS = 12 # 线程数，可根据CPU核心数调整

    # 1. 获取所有掩码路径
    mask_paths = sorted(glob.glob(os.path.join(MASK_DIR, "*.tiff")))
    id_list = [int(os.path.basename(p).replace(".tiff", "")) for p in mask_paths]
    
    print(f"Processing {len(mask_paths)} nuclei with {MAX_WORKERS} threads...")
    
    # 2. 生成斐波那契球面采样模板（考虑各向异性）
    ray_directions = fibonacci_sphere(n_samples=N_RAYS, anisotropy=ANISOTROPY)
    
    # 3. 多线程参数化采样所有细胞核
    correspondent_pcs = [None] * len(mask_paths)  # 预分配列表
    
    # 准备参数
    args_list = [(mask_path, obj_id, ray_directions) for mask_path, obj_id in zip(mask_paths, id_list)]
    
    # 使用线程池处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_idx = {executor.submit(process_single_mask, args): i for i, args in enumerate(args_list)}
        
        # 使用tqdm显示进度
        with tqdm(total=len(mask_paths), desc="Parametric Sampling") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    obj_id, pc = future.result()
                    if pc is not None:
                        correspondent_pcs[idx] = pc
                    else:
                        print(f"Failed to process mask {obj_id}")
                except Exception as e:
                    print(f"Exception occurred: {e}")
                pbar.update(1)
    
    # 过滤掉None值
    valid_pcs = [pc for pc in correspondent_pcs if pc is not None]
    valid_ids = [obj_id for obj_id, pc in zip(id_list, correspondent_pcs) if pc is not None]
    
    print(f"Successfully processed {len(valid_pcs)} out of {len(mask_paths)} nuclei")
    
    # 4. 广义Procrustes分析对齐
    print("Performing Generalized Procrustes Analysis...")
    aligned_pcs, mean_shape = generalized_procrustes(valid_pcs)
    
    # 5. 保存对齐后的点云
    original_points_dict = {}
    aligned_points_dict = {}
    
    for obj_id, pc, aligned_pc in zip(valid_ids, valid_pcs, aligned_pcs):
        original_points_dict[f"{obj_id}_orig"] = pc.astype(np.float32)
        aligned_points_dict[f"{obj_id}_aligned"] = aligned_pc.astype(np.float32)
    
    print(f"Saving aligned points to {OUT_FILE}...")
    np.savez(OUT_FILE, **original_points_dict, **aligned_points_dict)
    
    # 6. 创建形状模型
    print("Creating shape model...")
    aligned_points_flat = np.array([pc.ravel() for pc in aligned_pcs])
    
    from sklearn.decomposition import PCA
    n_components = min(30, len(aligned_pcs)-1)
    pca = PCA(n_components=n_components)
    pca.fit(aligned_points_flat)
    
    # 7. 保存形状模型
    print(f"Saving shape model to {SHAPE_MODEL_FILE}...")
    np.savez(SHAPE_MODEL_FILE,
             mean_shape=mean_shape.astype(np.float32),
             components=pca.components_.astype(np.float32),
             explained_variance=pca.explained_variance_.astype(np.float32),
             ray_directions=ray_directions.astype(np.float32))
    
    print("Done.")