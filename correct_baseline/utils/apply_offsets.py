import os
import json
import numpy as np
from skimage import io
import tifffile
from .realignment import apply_shift_with_padding, apply_shift_with_padding_seg

def load_offsets(json_path):
    """加载偏移信息"""
    with open(json_path, 'r') as f:
        return json.load(f)

def save_shifts(shifts, file_path, original_shape):
    """保存位移信息到JSON文件"""
    # 转换为可序列化的格式 (dy, dx)
    serializable_shifts = [(float(dy), float(dx)) for dy, dx in shifts]
    
    with open(file_path, 'w') as f:
        json.dump({
            "shifts": serializable_shifts,
            "original_shape": list(original_shape)  # 保存原始尺寸
        }, f)

def load_shifts(file_path):
    """从JSON文件加载位移信息"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    shifts = [(dy, dx) for dy, dx in data["shifts"]]
    original_shape = tuple(data["original_shape"])
    return shifts, original_shape

def restore_original_image(aligned_volume, shifts, original_shape):
    """
    将对齐后的图像恢复至原始状态
    
    Args:
        aligned_volume: 对齐后的图像体积 (Z, H, W)
        shifts: 每个切片的累积位移 (dy, dx) 列表
        original_shape: 原始图像形状 (Z, H, W)
    
    Returns:
        restored_volume: 恢复后的原始图像
    """
    # 1. 计算原始画布大小和填充
    all_shifts = np.array(shifts)
    min_y, min_x = np.floor(np.min(all_shifts, axis=0)).astype(int)
    pad_top = max(0, -min_y)
    pad_left = max(0, -min_x)
    
    # 2. 创建恢复后的体积
    restored_volume = np.zeros(original_shape, dtype=aligned_volume.dtype)
    
    # 3. 对每个切片恢复原始位置
    for z in range(original_shape[0]):
        dy, dx = shifts[z]
        dy_int = int(np.round(dy))
        dx_int = int(np.round(dx))
        
        # 从对齐图像中提取当前切片
        aligned_slice = aligned_volume[z]
        
        # 计算原始图像在填充画布中的位置
        y_start = pad_top + dy_int
        y_end = y_start + original_shape[1]
        x_start = pad_left + dx_int
        x_end = x_start + original_shape[2]
        
        # 提取原始图像区域
        # 确保坐标在有效范围内
        y_start = max(0, min(y_start, aligned_slice.shape[0] - 1))
        y_end = max(0, min(y_end, aligned_slice.shape[0]))
        x_start = max(0, min(x_start, aligned_slice.shape[1] - 1))
        x_end = max(0, min(x_end, aligned_slice.shape[1]))
        
        # 确保有效区域
        if y_end > y_start and x_end > x_start:
            original_region = aligned_slice[y_start:y_end, x_start:x_end]
            restored_volume[z] = original_region
    
    return restored_volume

def restore_original_segmentation(aligned_seg, shifts, original_shape):
    """
    将对齐后的分割恢复至原始状态
    
    Args:
        aligned_seg: 对齐后的分割体积 (Z, H, W)
        shifts: 每个切片的累积位移 (dy, dx) 列表
        original_shape: 原始分割形状 (Z, H, W)
    
    Returns:
        restored_seg: 恢复后的原始分割
    """
    # 1. 计算原始画布大小和填充
    all_shifts = np.array(shifts)
    min_y, min_x = np.floor(np.min(all_shifts, axis=0)).astype(int)
    pad_top = max(0, -min_y)
    pad_left = max(0, -min_x)
    
    # 2. 创建恢复后的体积
    restored_seg = np.zeros(original_shape, dtype=aligned_seg.dtype)
    
    # 3. 对每个切片恢复原始位置
    for z in range(original_shape[0]):
        dy, dx = shifts[z]
        dy_int = int(np.round(dy))
        dx_int = int(np.round(dx))
        
        # 从对齐分割中提取当前切片
        aligned_slice = aligned_seg[z]
        
        # 计算原始分割在填充画布中的位置
        y_start = pad_top + dy_int
        y_end = y_start + original_shape[1]
        x_start = pad_left + dx_int
        x_end = x_start + original_shape[2]
        
        # 提取原始分割区域
        # 确保坐标在有效范围内
        y_start = max(0, min(y_start, aligned_slice.shape[0] - 1))
        y_end = max(0, min(y_end, aligned_slice.shape[0]))
        x_start = max(0, min(x_start, aligned_slice.shape[1] - 1))
        x_end = max(0, min(x_end, aligned_slice.shape[1]))
        
        # 确保有效区域
        if y_end > y_start and x_end > x_start:
            original_region = aligned_slice[y_start:y_end, x_start:x_end]
            restored_seg[z] = original_region
    
    return restored_seg

def process_single_file(img_path, seg_path, offsets_json, output_dir):
    """
    处理单个文件：应用偏移并保存结果
    
    Args:
        img_path: 图像文件路径
        seg_path: 分割文件路径
        offsets_json: 偏移信息JSON文件路径
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取文件ID
    file_id = os.path.basename(img_path).split('_')[0]
    
    # 加载偏移信息
    offsets = load_offsets(offsets_json)
    if file_id not in offsets:
        raise ValueError(f"No offset information found for file {file_id}")
    
    # 读取数据
    img = io.imread(img_path)
    seg = io.imread(seg_path)
    original_shape = img.shape  # 保存原始尺寸
    
    # 打印原始数据信息
    print(f"Original image shape: {img.shape}, dtype: {img.dtype}")
    print(f"Original segmentation shape: {seg.shape}, dtype: {seg.dtype}")
    
    # 获取偏移信息
    offset_info = offsets[str(file_id)]
    relative_shifts = offset_info['relative_shifts']
    bad_slices = offset_info['bad_slices']
    
    # 转换relative_shifts为flow_volume格式
    flow_volume = []
    for shift_info in relative_shifts:
        if shift_info['status'] == 'skip':
            flow_volume.append({'skip': True})
        else:
            flow_volume.append({
                'dx': shift_info['dx'],
                'dy': shift_info['dy'],
                'quality': shift_info['quality'],
                'ref_index': shift_info['from_slice'],
                'mov_index': shift_info['to_slice']
            })
    
    # 使用realignment.py中的函数应用偏移
    aligned_img, shifts, valid_indices = apply_shift_with_padding(img, flow_volume, bad_slices)
    aligned_seg = apply_shift_with_padding_seg(seg, shifts)
    
    # 保存对齐后的结果
    aligned_img_path = os.path.join(output_dir, f'{file_id}_aligned.tif')
    aligned_seg_path = os.path.join(output_dir, f'{file_id}_aligned_seg.tif')
    
    tifffile.imwrite(aligned_img_path, aligned_img, compression='zlib')
    tifffile.imwrite(aligned_seg_path, aligned_seg, compression='zlib')
    
    # 打印对齐后数据信息
    print(f"Aligned image shape: {aligned_img.shape}, dtype: {aligned_img.dtype}")
    print(f"Aligned segmentation shape: {aligned_seg.shape}, dtype: {aligned_seg.dtype}")
    
    # 保存位移信息
    shifts_path = os.path.join(output_dir, f'{file_id}_shifts.json')
    save_shifts(shifts, shifts_path, original_shape)
    
    # 执行恢复操作
    # 加载对齐后的数据
    aligned_img_restore = io.imread(aligned_img_path)
    aligned_seg_restore = io.imread(aligned_seg_path)
    
    # 加载位移信息
    shifts_restore, original_shape_restore = load_shifts(shifts_path)
    
    # 恢复原始图像
    restored_img = restore_original_image(aligned_img_restore, shifts_restore, original_shape_restore)
    restored_seg = restore_original_segmentation(aligned_seg_restore, shifts_restore, original_shape_restore)
    
    # 打印恢复后数据信息
    print(f"Restored image shape: {restored_img.shape}, dtype: {restored_img.dtype}")
    print(f"Restored segmentation shape: {restored_seg.shape}, dtype: {restored_seg.dtype}")
    
    # 保存恢复结果
    restored_img_path = os.path.join(output_dir, f'{file_id}_restored.tif')
    restored_seg_path = os.path.join(output_dir, f'{file_id}_restored_seg.tif')
    
    tifffile.imwrite(restored_img_path, restored_img, compression='zlib')
    tifffile.imwrite(restored_seg_path, restored_seg, compression='zlib')
    
    return {
        'file_id': file_id,
        'aligned_img': aligned_img_path,
        'aligned_seg': aligned_seg_path,
        'restored_img': restored_img_path,
        'restored_seg': restored_seg_path,
        'shifts': shifts_path
    }

if __name__ == "__main__":
    # 设置路径
    img_path = r"D:\paper\fafb_process\tmp\8533661_img.tiff"  # 输入图像路径
    seg_path = r"D:\paper\fafb_process\tmp\8533661_mis.tiff"  # 输入分割路径
    offsets_json = r"D:\paper\fafb_data\merge_error\merge_error_correct_6.14\slice_offsets.json"  # 偏移信息
    output_dir = r"D:\paper\fafb_process\tmp\aligned"  # 输出目录
    
    # 处理文件
    try:
        result = process_single_file(img_path, seg_path, offsets_json, output_dir)
        print("\nProcessing completed successfully!")
        print(f"File ID: {result['file_id']}")
        print(f"Aligned image: {result['aligned_img']}")
        print(f"Aligned segmentation: {result['aligned_seg']}")
        print(f"Restored image: {result['restored_img']}")
        print(f"Restored segmentation: {result['restored_seg']}")
        print(f"Shifts saved to: {result['shifts']}")
    except Exception as e:
        print(f"Error: {str(e)}")