from ZMPY3D_PT import get_global_parameter, calculate_bbox_moment, \
    get_bbox_moment_xyz_sample, calculate_molecular_radius, calculate_bbox_moment_2_zm, get_3dzd_121_descriptor
import torch
import pickle
import os


def load_zernike_cache(max_order=20, device='cuda'):
    cache_path = os.path.join('/nvme2/mingzhi/NucCorr/NucDet', f'ZMPY3D_PT/cache_data/LogG_CLMCache_MaxOrder{max_order:02d}.pkl')
    with open(cache_path, 'rb') as file:
        CachePKL = pickle.load(file)
    cache = {
        'GCache_pqr_linear': torch.tensor(CachePKL['GCache_pqr_linear'], device=device),
        'GCache_complex': torch.tensor(CachePKL['GCache_complex'], device=device),
        'GCache_complex_index': torch.tensor(CachePKL['GCache_complex_index'], device=device),
        'CLMCache3D': torch.tensor(CachePKL['CLMCache3D'], dtype=torch.complex128, device=device)
    }
    return cache


def compute_zernike_descriptor_from_tensor(volume_tensor, cache, max_order=20):
    
    device = volume_tensor.device
    Param = get_global_parameter()
    GCache_pqr_linear = cache['GCache_pqr_linear']
    GCache_complex = cache['GCache_complex']
    GCache_complex_index = cache['GCache_complex_index']
    CLMCache3D = cache['CLMCache3D']

    # 1. 计算质心
    dims = volume_tensor.shape
    X = torch.arange(0, dims[0] + 1, dtype=torch.float64, device=device)
    Y = torch.arange(0, dims[1] + 1, dtype=torch.float64, device=device)
    Z = torch.arange(0, dims[2] + 1, dtype=torch.float64, device=device)
    order_tensor = torch.tensor(max_order, dtype=torch.int64, device=device)

    mass, center, _ = calculate_bbox_moment(volume_tensor, 1, X, Y, Z)
    
    # 2. 分子半径
    avg_radius, max_radius = calculate_molecular_radius(
        volume_tensor, center, mass, torch.tensor(Param['default_radius_multiplier'], dtype=torch.float64, device=device)
    )


    # 3. 球坐标采样
    sX, sY, sZ = get_bbox_moment_xyz_sample(center, avg_radius, dims)

    # 4. 球面几何矩
    _, _, sphere_moments = calculate_bbox_moment(volume_tensor, order_tensor, sX, sY, sZ)

    # 5. Zernike Moment 投影
    zernike_scaled, _ = calculate_bbox_moment_2_zm(
        order_tensor,
        GCache_complex,
        GCache_pqr_linear,
        GCache_complex_index,
        CLMCache3D,
        sphere_moments
    )

    # 6. 121维 Zernike Descriptor
    descriptor = get_3dzd_121_descriptor(zernike_scaled)  # shape: (11,11) upper triangle
    descriptor = torch.flatten(descriptor[~torch.isnan(descriptor)])

    return descriptor  # shape: (121,)

""" volume = torch.from_numpy(obj1).to(torch.float64).cuda()
volume[volume==1.0] = 1
zernike_feature = compute_zernike_descriptor_from_tensor(volume)
volume2 = torch.from_numpy(obj2).to(torch.float64).cuda()
zernike_feature2 = compute_zernike_descriptor_from_tensor(volume2)
print(zernike_feature2)
print(zernike_feature)
cos_sim = torch.dot(zernike_feature2, zernike_feature) / (zernike_feature2.norm() * zernike_feature.norm())
print("Cosine similarity:", cos_sim.item()) """