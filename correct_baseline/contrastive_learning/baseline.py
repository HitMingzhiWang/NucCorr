import os
import sys
import numpy as np
from skimage import io
from tqdm import tqdm
from scipy import ndimage
sys.path.append('/nvme2/mingzhi/NucCorr')
from correct_baseline.utils.error_helper import *
from correct_baseline.utils.apply_offsets import load_offsets, restore_original_segmentation
from correct_baseline.utils.realignment import apply_shift_with_padding, apply_shift_with_padding_seg
from correct_baseline.contrastive_learning.pointNet.test import test_ids_same_object

def merge_error_correction(img, seg, file_id, offsets, output_dir=None):
    """
    对原始 segmentation 做 realignment + merge correction。
    返回 merge 后 segmentation，以及来自 ID=1 的后代 ID 列表。
    """
    original_id1_mask = (seg == 1).copy()
    original_shape = img.shape

    print("Step 1: Processing bad slices...")
    if offsets is not None and str(file_id) in offsets:
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
    else:
        print(f"{file_id} not in offsets or offsets is None, skipping alignment...")
        aligned_img = img
        aligned_seg = seg
        shifts = None

    print("Step 4: Performing merge correction...")
    # baseline: 不做merge修正，直接用process_all_components_with_safe_merge
    corrected_seg = process_all_components_with_safe_merge(aligned_seg, iteration=2, erosion_shape='ball')
    print(np.unique(corrected_seg))
    # 提取 merge 后 ID=1 演化出来的区域
    id1_descendant_mask = (corrected_seg > 0) & (aligned_seg==1)
    id1_related_ids = np.unique(corrected_seg[id1_descendant_mask])
    id1_related_ids = id1_related_ids[id1_related_ids != 0]
    print(f"ID=1 maps to {len(id1_related_ids)} new IDs after merge correction: {id1_related_ids}")
    return aligned_img, corrected_seg, shifts, original_shape, id1_related_ids.tolist()

def split_error_correction(seg, volume, id1_related_ids, cnn_model, pointnet_model, device='cuda', n_points=2048):
    """
    将所有与 id1_related_ids 匹配的候选 mask 的 ID 重设为目标 ID。
    其他区域设为 0。
    """
    print("Step 5: Performing match-based ID correction...")

    unique_ids = np.unique(seg)
    unique_ids = unique_ids[unique_ids != 0]

    id_to_mask = {}
    for label_id in unique_ids:
        id_to_mask[label_id] = (seg == label_id)

    matched_ids = set(id1_related_ids)
    final_seg = np.zeros_like(seg)

    for main_id in id1_related_ids:
        if main_id not in unique_ids:
            continue

        # 自己保留
        final_seg[id_to_mask[main_id]] = main_id

        for cand_id in unique_ids:
            if cand_id == main_id or cand_id in matched_ids:
                continue

            try:
                result = test_ids_same_object(volume, seg, main_id, cand_id, cnn_model, pointnet_model, device=device)
                if result is not None and result['is_same_object']:
                    matched_ids.add(cand_id)
                    final_seg[id_to_mask[cand_id]] = main_id  # 修改为当前目标ID
            except Exception as e:
                print(f"Error testing ids {main_id}, {cand_id}: {e}")
                continue

    print(f"Matched {len(matched_ids)} IDs (including original targets).")
    return final_seg

if __name__ == "__main__":
    # === 路径配置 ===
    img_path = "/nvme2/mingzhi/NucCorr/NucCorrData/split_error/img/560693_img.tiff"
    seg_path = "/nvme2/mingzhi/NucCorr/NucCorrData/split_error/match_seg/560693.tiff"
    offsets_json = "/nvme2/mingzhi/NucCorr/NucCorrData/split_error/slice_offsets.json"
    output_dir = "tmp_results"
    os.makedirs(output_dir, exist_ok=True)

    # === 读取数据 ===
    img = io.imread(img_path)
    seg = io.imread(seg_path)
    offsets = load_offsets(offsets_json)
    file_id = os.path.basename(img_path).split('_')[0]

    # 加载3D CNN模型
    import torch
    import importlib.util
    device = 'cuda'
    spec = importlib.util.spec_from_file_location(
        "model_3DCNN", 
        "/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/3DCNN/model/3DCNN.py"
    )
    model_3DCNN = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_3DCNN)
    EmbedNetUNet = model_3DCNN.EmbedNetUNet
    
    cnn_model = EmbedNetUNet().to(device)  
    cnn_checkpoint_path = "/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/3DCNN/logs/train_20250728-182654/checkpoint_best.pth"
    if os.path.exists(cnn_checkpoint_path):
        checkpoint = torch.load(cnn_checkpoint_path, map_location=device)
        cnn_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 加载3D CNN权重: {cnn_checkpoint_path}")
    else:
        print(f"✗ 未找到3D CNN权重: {cnn_checkpoint_path}")
    cnn_model.eval()
    
    # 加载PointNet2模型
    from correct_baseline.contrastive_learning.pointNet.model.pointnet2 import PointNet2Contrastive
    pointnet_model = PointNet2Contrastive(
        input_dim=3,
        embed_dim=16,
        num_classes=1,
        use_embeddings=True
    ).to(device)
    pointnet_checkpoint_path = "/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/pointNet/checkpoints/best_model.pth"
    if os.path.exists(pointnet_checkpoint_path):
        checkpoint = torch.load(pointnet_checkpoint_path, map_location=device, weights_only=False)
        pointnet_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 加载PointNet2权重: {pointnet_checkpoint_path}")
    else:
        print(f"✗ 未找到PointNet2权重: {pointnet_checkpoint_path}")
    pointnet_model.eval()

    try:
        aligned_img, corrected_seg, shifts, original_shape, id1_related_ids = merge_error_correction(
            img, seg, file_id, offsets, output_dir
        )
        final_seg = split_error_correction(corrected_seg, aligned_img, id1_related_ids, cnn_model, pointnet_model, device=device)
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