import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
import os
import tifffile
from scipy.ndimage import gaussian_filter
import vedo
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def load_masks(mask_dir):
    """加载所有3D mask文件"""
    masks = []
    for fname in os.listdir(mask_dir):
        if fname.endswith('.tif') or fname.endswith('.tiff'):
            mask = tifffile.imread(os.path.join(mask_dir, fname))
            masks.append(mask)
    return masks

def extract_single_nuclei(mask_3d, min_volume=1000):
    """从3D mask中提取单个细胞核实例"""
    labeled = label(mask_3d)
    props = regionprops(labeled)
    nuclei_list = []
    for prop in props:
        if prop.area < min_volume:
            continue  # 过滤小碎片
        minr, minc, minz, maxr, maxc, maxz = prop.bbox
        nucleus = labeled[minr:maxr, minc:maxc, minz:maxz] == prop.label
        nuclei_list.append(nucleus.astype(np.uint8))
    return nuclei_list

def compute_sdf(mask, normalize=True):
    """计算二值mask的符号距离函数(SDF)"""
    inner_dist = distance_transform_edt(mask)
    outer_dist = distance_transform_edt(np.logical_not(mask))
    sdf = inner_dist - outer_dist
    
    if normalize:
        # 方法1: 除以最大绝对值进行归一化到[-1, 1]
        max_abs = np.max(np.abs(sdf))
        if max_abs > 0:
            sdf = sdf / max_abs
        
        # 方法2: 如果你想用标准化 (零均值，单位方差)，可以用下面的代码替换上面的
        # sdf = (sdf - np.mean(sdf)) / (np.std(sdf) + 1e-8)
        
        # 方法3: 如果你想用min-max归一化到[0, 1]，可以用下面的代码
        # sdf_min, sdf_max = np.min(sdf), np.max(sdf)
        # if sdf_max > sdf_min:
        #     sdf = (sdf - sdf_min) / (sdf_max - sdf_min)
    
    return sdf

def resize_and_pad_nucleus(nucleus_mask, target_size=(64, 64, 64)):
    """调整nucleus mask到目标尺寸（无需质心对齐）"""
    # 检查输入mask是否为空
    if nucleus_mask.size == 0 or np.sum(nucleus_mask) == 0:
        print("Warning: Empty nucleus mask detected, skipping...")
        return np.zeros(target_size, dtype=nucleus_mask.dtype)
    
    current_shape = np.array(nucleus_mask.shape)
    target_shape = np.array(target_size)
    
    # 情况1：如果当前形状等于目标形状，直接返回
    if np.array_equal(current_shape, target_shape):
        return nucleus_mask
    
    # 情况2：如果当前形状小于等于目标形状，进行零填充
    if np.all(current_shape <= target_shape):
        padded = np.zeros(target_size, dtype=nucleus_mask.dtype)
        # 计算填充的起始位置（居中放置）
        start_pos = (target_shape - current_shape) // 2
        end_pos = start_pos + current_shape
        
        # 创建切片
        slices = tuple(slice(start_pos[i], end_pos[i]) for i in range(3))
        padded[slices] = nucleus_mask
        return padded
    
    # 情况3：如果当前形状大于目标形状，需要裁剪
    elif np.all(current_shape >= target_shape):
        # 计算裁剪的起始位置（居中裁剪）
        start_pos = (current_shape - target_shape) // 2
        end_pos = start_pos + target_shape
        
        # 创建切片
        slices = tuple(slice(start_pos[i], end_pos[i]) for i in range(3))
        return nucleus_mask[slices]
    
    # 情况4：混合情况，某些维度需要填充，某些需要裁剪
    else:
        # 先处理需要裁剪的维度
        cropped = nucleus_mask
        crop_slices = []
        
        for i in range(3):
            if current_shape[i] > target_shape[i]:
                # 需要裁剪
                start_pos = (current_shape[i] - target_shape[i]) // 2
                end_pos = start_pos + target_shape[i]
                crop_slices.append(slice(start_pos, end_pos))
            else:
                crop_slices.append(slice(None))
        
        cropped = nucleus_mask[tuple(crop_slices)]
        
        # 再处理需要填充的维度
        cropped_shape = np.array(cropped.shape)
        if np.array_equal(cropped_shape, target_shape):
            return cropped
        
        padded = np.zeros(target_size, dtype=nucleus_mask.dtype)
        pad_slices = []
        
        for i in range(3):
            if cropped_shape[i] < target_shape[i]:
                # 需要填充
                start_pos = (target_shape[i] - cropped_shape[i]) // 2
                end_pos = start_pos + cropped_shape[i]
                pad_slices.append(slice(start_pos, end_pos))
            else:
                pad_slices.append(slice(None))
        
        padded[tuple(pad_slices)] = cropped
        return padded

def create_avg_sdf_template(nuclei_list, target_size=(64, 64, 64), sigma=1.0, max_workers=8):
    """创建平均SDF模板（多线程加速）"""
    aligned_sdfs = []
    
    def process_nucleus(nucleus_mask):
        try:
            resized = resize_and_pad_nucleus(nucleus_mask, target_size)
            # 检查resized是否为空
            if np.sum(resized) == 0:
                return None
            sdf = compute_sdf(resized)
            return sdf
        except Exception as e:
            print(f"Error processing nucleus: {e}")
            return None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_nucleus, nucleus_mask) for nucleus_mask in nuclei_list]
        for f in tqdm(as_completed(futures), total=len(futures), desc='SDF计算'):
            result = f.result()
            if result is not None:
                aligned_sdfs.append(result)
    
    if len(aligned_sdfs) == 0:
        raise ValueError("No valid nuclei found for SDF template creation")
    
    print(f"Successfully processed {len(aligned_sdfs)} out of {len(nuclei_list)} nuclei")
    avg_sdf = np.mean(aligned_sdfs, axis=0)
    avg_sdf = gaussian_filter(avg_sdf, sigma=sigma)
    return avg_sdf

""" def visualize_sdf(sdf, threshold=0):
    mesh = vedo.Volume(sdf).isosurface(threshold)
    mesh.cmap('coolwarm', on='points', vmin=np.min(sdf), vmax=np.max(sdf))
    mesh.addScalarBar(title='Signed Distance')
    axes = vedo.Axes(mesh)
    plt = vedo.Plotter(offscreen=True, bg='white')
    plt.show(mesh, axes=7)
    plt.screenshot('output.png')
    plt.close() """

# ===== 主流程 =====
if __name__ == "__main__":
    masks = load_masks("/nvme2/mingzhi/NucCorr/NucCorrData/normal_nuclei/normal_nuclei/seg")
    all_nuclei = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(extract_single_nuclei, mask) for mask in masks]
        for f in tqdm(as_completed(futures), total=len(futures), desc='提取单核'):
            all_nuclei.extend(f.result())
    
    print(f"提取到 {len(all_nuclei)} 个细胞核实例")
    
    # 过滤掉空的nucleus mask
    valid_nuclei = [n for n in all_nuclei if n.size > 0 and np.sum(n) > 0]
    print(f"有效细胞核实例: {len(valid_nuclei)}")
    
    if len(valid_nuclei) == 0:
        print("错误: 没有找到有效的细胞核实例")
        exit(1)
    
    template_size = (128, 128, 128)
    avg_sdf = create_avg_sdf_template(valid_nuclei, target_size=template_size, sigma=1.2, max_workers=8)
    
    np.save("/nvme2/mingzhi/NucCorr/correct_baseline/STN/nuclei_avg_sdf_template.npy", avg_sdf)
    #visualize_sdf(avg_sdf, threshold=0)
    print("平均SDF模板创建完成！")