import os
import sys
import torch
import numpy as np
from skimage import io
from utils.error_helper import *
from utils.apply_offsets import load_offsets, restore_original_segmentation, restore_original_image
from utils.realignment import apply_shift_with_padding, apply_shift_with_padding_seg
from utils.error_helper import process_all_components_with_safe_merge
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from complete_model.model.unet import get_model

def split_error_correction(aligned_img, corrected_seg, model_path, device="cuda:2"):
    """
    处理split error:
    1. 对每个id提取128x128x128的patch
    2. 使用UNet预测完整mask
    3. 使用NMS策略合并重叠mask
    
    Args:
        aligned_img: 对齐后的图像
        corrected_seg: 对齐后的分割
        model_path: UNet模型路径
        device: 使用的设备
    
    Returns:
        final_seg: 处理后的分割结果
    """
    print("Step 1: Loading UNet model...")
    model = get_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 获取所有唯一的id（除了0）
    unique_ids = np.unique(corrected_seg)[1:]
    print(f"Found {len(unique_ids)} unique labels")
    
    # 存储每个id的完整mask
    complete_masks = {}
    
    # 对每个id进行处理
    for label_id in tqdm(unique_ids, desc="SplitError UNet", ncols=80):
        # 获取当前id的mask
        current_mask = (corrected_seg == label_id)
        
        # 找到mask的中心点
        z_indices, y_indices, x_indices = np.where(current_mask)
        center_z = int(np.mean(z_indices))
        center_y = int(np.mean(y_indices))
        center_x = int(np.mean(x_indices))
        
        # 计算patch的边界
        z_start = max(0, center_z - 64)
        z_end = min(aligned_img.shape[0], center_z + 64)
        y_start = max(0, center_y - 64)
        y_end = min(aligned_img.shape[1], center_y + 64)
        x_start = max(0, center_x - 64)
        x_end = min(aligned_img.shape[2], center_x + 64)
        
        # 提取patch
        img_patch = aligned_img[z_start:z_end, y_start:y_end, x_start:x_end]
        seg_patch = current_mask[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # 创建128x128x128的patch（用0填充）
        full_img_patch = np.zeros((128, 128, 128), dtype=np.float32)
        full_seg_patch = np.zeros((128, 128, 128), dtype=np.float32)
        
        # 将提取的patch放入完整patch中
        patch_shape = img_patch.shape
        full_img_patch[:patch_shape[0], :patch_shape[1], :patch_shape[2]] = img_patch
        full_seg_patch[:patch_shape[0], :patch_shape[1], :patch_shape[2]] = seg_patch
        
        # 转换为tensor
        img_tensor = torch.from_numpy(full_img_patch).float().unsqueeze(0).unsqueeze(0) / 255.0
        seg_tensor = torch.from_numpy(full_seg_patch).float().unsqueeze(0).unsqueeze(0)
        
        # 移动到GPU
        img_tensor = img_tensor.to(device)
        seg_tensor = seg_tensor.to(device)
        
        # 模型预测
        with torch.no_grad():
            x = torch.cat([seg_tensor, img_tensor], dim=1)
            output = model(x)
            pred = (torch.sigmoid(output) > 0.5).float()
            
            # 将预测结果放回原始位置
            pred_mask = pred.squeeze().cpu().numpy()
            complete_mask = np.zeros_like(corrected_seg, dtype=bool)
            complete_mask[z_start:z_end, y_start:y_end, x_start:x_end] = pred_mask[:patch_shape[0], :patch_shape[1], :patch_shape[2]]
            
            complete_masks[label_id] = complete_mask
    
    print("Step 2: Applying NMS to masks...")
    # 按体素大小降序排序ID
    sorted_ids = sorted(unique_ids, key=lambda id: complete_masks[id].sum(), reverse=True)
    
    keep_ids = []  # 保留的ID列表
    suppressed = set()  # 被抑制的ID集合
    
    # 遍历所有ID（按大小降序）
    for i, current_id in enumerate(sorted_ids):
        if current_id in suppressed:
            continue
            
        keep_ids.append(current_id)
        current_mask = complete_masks[current_id]
        
        # 检查后续所有ID
        for other_id in sorted_ids[i+1:]:
            if other_id in suppressed:
                continue
                
            other_mask = complete_masks[other_id]
            intersection = np.logical_and(current_mask, other_mask).sum()
            
            # 如果两个掩码不相交，跳过
            if intersection == 0:
                continue
                
            # 计算IoU和交集比例
            union = np.logical_or(current_mask, other_mask).sum()
            iou = intersection / union
            ratio_current = intersection / current_mask.sum()
            ratio_other = intersection / other_mask.sum()
            
            # 检查抑制条件：IoU > 0.5 或 交集占任一掩码50%以上
            if iou > 0.5 or ratio_current > 0.5 or ratio_other > 0.5:
                suppressed.add(other_id)
    
    print(f"Kept {len(keep_ids)} masks after suppressing {len(suppressed)} masks")
    
    print("Step 3: Creating final segmentation...")
    final_seg = np.zeros_like(corrected_seg)
    for new_label, orig_id in enumerate(keep_ids, 1):
        mask = complete_masks[orig_id]
        to_assign = mask & (final_seg == 0)
        final_seg[to_assign] = new_label
    
    return final_seg

def merge_error_correction(img, seg, file_id, offsets, output_dir):
    if str(file_id) not in offsets:
        print(f"{file_id} not in offsets, skipping...")
        return img, seg, None, img.shape
    original_shape = img.shape
    print("Step 1: Processing bad slices...")
    if str(file_id) in offsets:
        bad_slices = offsets[str(file_id)]['bad_slices']
        print(f"Found {len(bad_slices)} bad slices")
        for slice_idx in bad_slices:
            seg[slice_idx] = 0
    else:
        print("No bad slices information found")
    
    # 准备对齐数据
    print("Step 2: Preparing alignment data...")
    offset_info = offsets[str(file_id)]
    relative_shifts = offset_info['relative_shifts']
    bad_slices = offset_info['bad_slices']
    
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
    
    # 执行对齐
    print("Step 3: Performing alignment...")
    aligned_img, shifts, valid_indices = apply_shift_with_padding(img, flow_volume, bad_slices)
    aligned_seg = apply_shift_with_padding_seg(seg, shifts)
    
    # 执行merge correction
    print("Step 4: Performing merge correction...")
    #corrected_seg = detect_and_correct_merge_wmz(aligned_seg, iteration=3)
    corrected_seg = process_all_components_with_safe_merge(aligned_seg,iteration=2,erosion_shape='ball')
    return aligned_img, corrected_seg, shifts, original_shape




if __name__ == "__main__":
    # 设置路径
    img_path = "/nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14/img/1958996_img.tiff"
    seg_path = "/nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14/seg/1958996.tiff"
    offsets_json = "/nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14/slice_offsets.json"
    model_path = "/nvme2/mingzhi/NucCorr/complete_model/checkpoints/best_model_epoch_bce.pth"
    output_dir = "/nvme2/mingzhi/NucCorr/correct_baseline/tmp_results"
    
    # 读取数据
    img = io.imread(img_path)
    seg = io.imread(seg_path)
    offsets = load_offsets(offsets_json)
    
    # 执行处理流程
    try:
        file_id = os.path.basename(img_path).split('_')[0]
        aligned_img, corrected_seg, shifts, original_shape = merge_error_correction(img, seg, file_id, offsets, output_dir)
        
        # 执行split error correction
        print("Step 5: Performing split error correction...")
        final_seg = split_error_correction(aligned_img, corrected_seg, model_path)
        #final_seg = corrected_seg
        # 执行restore操作
        print("Step 6: Restoring segmentation to original space...")
        if shifts is not None:
            restored_seg = restore_original_segmentation(final_seg, shifts, original_shape)
        else:
            print("No shifts found, skipping restore step.")
            restored_seg = final_seg
        
        # 保存结果
        final_output_dir = os.path.join(output_dir, 'final_results')
        os.makedirs(final_output_dir, exist_ok=True)
        aligned_img_path = os.path.join(final_output_dir, f'{file_id}_aligned_img.tiff')
        final_seg_path = os.path.join(final_output_dir, f'{file_id}_final_seg.tiff')
        io.imsave(final_seg_path, final_seg.astype(np.uint8))
        io.imsave(aligned_img_path,aligned_img)
        
        print("\nProcessing completed successfully!")
        print(f"Results saved to: {final_output_dir}")
    except Exception as e:
        print(f"Error: {str(e)}")

