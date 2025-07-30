import os
import sys
import torch
import numpy as np
import math
from skimage import io
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/nvme2/mingzhi/NucCorr')

from correct_baseline.utils.error_helper import *
from correct_baseline.utils.apply_offsets import load_offsets, restore_original_segmentation
from correct_baseline.utils.realignment import apply_shift_with_padding, apply_shift_with_padding_seg


def compute_centroid(mask):
    indices = np.argwhere(mask)
    return np.mean(indices, axis=0) if len(indices) > 0 else np.array([0, 0, 0])

def is_match(center1, center2, max_distance=20, max_angle_deg=30):
    vec = center2 - center1
    distance = np.linalg.norm(vec)
    if distance == 0:
        return True
    angle_rad = math.acos(abs(vec[0]) / distance)
    angle_deg = math.degrees(angle_rad)
    return (distance < max_distance) and (angle_deg < max_angle_deg)


def merge_error_correction(img, seg, file_id, offsets, output_dir):
    """
    对原始 segmentation 做 realignment + merge correction。
    返回 merge 后 segmentation，以及来自 ID=1 的后代 ID 列表。
    """
    original_id1_mask = (seg == 1).copy()
    original_shape = img.shape

    print("Step 1: Processing bad slices...")
    if str(file_id) not in offsets:
        print(f"{file_id} not in offsets, skipping...")
        aligned_img = img
        aligned_seg = seg
        shifts = None
        print("Step 4: Performing merge correction...")
        corrected_seg = process_all_components_with_safe_merge(aligned_seg, iteration=2, erosion_shape='ball')
        print(np.unique(corrected_seg))
        # 提取 merge 后 ID=1 演化出来的区域
        id1_descendant_mask = (corrected_seg > 0) & (aligned_seg==1)
        id1_related_ids = np.unique(corrected_seg[id1_descendant_mask])
        id1_related_ids = id1_related_ids[id1_related_ids != 0]
        print(f"ID=1 maps to {len(id1_related_ids)} new IDs after merge correction: {id1_related_ids}")
        return aligned_img, corrected_seg, None, original_shape, id1_related_ids.tolist()

    bad_slices = offsets[str(file_id)]['bad_slices']
    for slice_idx in bad_slices:
        seg[slice_idx] = 0

    print("Step 2: Preparing alignment data...")
    relative_shifts = offsets[str(file_id)]['relative_shifts']
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

    print("Step 3: Performing alignment...")
    aligned_img, shifts, _ = apply_shift_with_padding(img, flow_volume, bad_slices)
    aligned_seg = apply_shift_with_padding_seg(seg, shifts)
    print("Step 4: Performing merge correction...")
    corrected_seg = process_all_components_with_safe_merge(aligned_seg, iteration=2, erosion_shape='ball')
    print(np.unique(corrected_seg))
    # 提取 merge 后 ID=1 演化出来的区域
    id1_descendant_mask = (corrected_seg > 0) & (aligned_seg==1)
    id1_related_ids = np.unique(corrected_seg[id1_descendant_mask])
    id1_related_ids = id1_related_ids[id1_related_ids != 0]

    print(f"ID=1 maps to {len(id1_related_ids)} new IDs after merge correction: {id1_related_ids}")
    return aligned_img, corrected_seg, shifts, original_shape, id1_related_ids.tolist()


def split_error_correction(corrected_seg, id1_related_ids):
    """
    将所有与 id1_related_ids 匹配的候选 mask 的 ID 重设为目标 ID。
    其他区域设为 0。
    """
    print("Step 5: Performing match-based ID correction...")

    unique_ids = np.unique(corrected_seg)
    unique_ids = unique_ids[unique_ids != 0]

    id_to_mask = {}
    id_to_center = {}

    for label_id in unique_ids:
        mask = (corrected_seg == label_id)
        id_to_mask[label_id] = mask
        id_to_center[label_id] = compute_centroid(mask)

    matched_ids = set(id1_related_ids)
    final_seg = np.zeros_like(corrected_seg)

    for main_id in id1_related_ids:
        main_center = id_to_center.get(main_id)
        if main_center is None:
            continue

        final_seg[id_to_mask[main_id]] = main_id  # 自己保留

        for cand_id in unique_ids:
            if cand_id == main_id or cand_id in matched_ids:
                continue
            cand_center = id_to_center[cand_id]

            if is_match(main_center, cand_center):
                matched_ids.add(cand_id)
                final_seg[id_to_mask[cand_id]] = main_id  # 修改为当前目标ID

    print(f"Matched {len(matched_ids)} IDs (including original targets).")
    return final_seg


if __name__ == "__main__":
    # === 路径配置 ===
    img_path = "/nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14/img/1955387_img.tiff"
    seg_path = "/nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14/match_seg/1955387.tiff"
    offsets_json = "/nvme2/mingzhi/NucCorr/NucCorrData/split_error/slice_offsets.json"
    output_dir = "tmp_results"
    os.makedirs(output_dir, exist_ok=True)

    # === 读取数据 ===
    img = io.imread(img_path)
    seg = io.imread(seg_path)
    offsets = load_offsets(offsets_json)
    file_id = os.path.basename(img_path).split('_')[0]

    try:
        aligned_img, corrected_seg, shifts, original_shape, id1_related_ids = merge_error_correction(
            img, seg, file_id, offsets, output_dir
        )

        final_seg = split_error_correction(corrected_seg, id1_related_ids)

        print("Step 6: Restoring to original space...")
        if shifts is not None:
            restored_seg = restore_original_segmentation(final_seg, shifts, original_shape)
        else:
            restored_seg = final_seg

        final_output_dir = os.path.join(output_dir, 'final_results')
        os.makedirs(final_output_dir, exist_ok=True)
        final_seg_path = os.path.join(final_output_dir, f'{file_id}_final_seg.tiff')
        io.imsave(final_seg_path, restored_seg.astype(np.uint8))

        print("\n✅ Processing completed successfully!")
        print(f"➡️ Final results saved to: {final_seg_path}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
