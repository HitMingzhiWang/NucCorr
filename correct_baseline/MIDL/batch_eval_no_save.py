import os
import re
import numpy as np
from tifffile import imread as tiff_imread
from tqdm import tqdm
import sys
import json
from multiprocessing import Pool, cpu_count
import argparse
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/nvme2/mingzhi/NucCorr')

from correct_baseline.utils.apply_offsets import load_offsets, restore_original_segmentation
from correct_baseline.utils.evaluate import match_instances, compute_metrics
from correct_baseline.MIDL.baseline import merge_error_correction, split_error_correction
from model.point_net import PointNet2Classification
import torch

def evaluate_pair(truth, pred, iou_threshold=0.8):
    truth_labels = truth.astype(np.int32)
    pred_labels = pred.astype(np.int32)
    tp, fp, fn = match_instances(truth_labels, pred_labels, iou_threshold)
    precision, recall, f1 = compute_metrics(tp, fp, fn)
    return {
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": precision, "Recall": recall, "F1": f1
    }

def process_and_eval(seg_path, img_path, correct_path, offsets, model, device, iou_threshold=0.8):
    file_id = re.findall(r'\d+', os.path.basename(seg_path))[0]

    img = tiff_imread(img_path)
    seg = tiff_imread(seg_path)
    correct = tiff_imread(correct_path)

    aligned_img, corrected_seg, shifts, original_shape, id1_related_ids = merge_error_correction(img, seg, file_id, offsets, None)
    # split_error_correction 需要传入 model/device
    final_seg = split_error_correction(corrected_seg, id1_related_ids, model, device=device)

    if shifts is not None:
        restored_seg = restore_original_segmentation(final_seg, shifts, original_shape)
    else:
        restored_seg = final_seg

    metrics = evaluate_pair(correct, restored_seg, iou_threshold)
    metrics["file_id"] = file_id
    return metrics

def worker(task):
    try:
        seg_path, img_path, correct_path, offsets, model, device, iou_threshold = task
        if not os.path.exists(img_path) or not os.path.exists(correct_path):
            return None
        return process_and_eval(seg_path, img_path, correct_path, offsets, model, device, iou_threshold)
    except Exception as e:
        print(f"[ERROR] {seg_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--seg_filter', type=str, default='all', help="ms, not_ms, or all")
    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of worker processes')
    args = parser.parse_args()

    base_dir = args.base_dir
    seg_dir = os.path.join(base_dir, "match_seg")
    img_dir = os.path.join(base_dir, "img")
    correct_dir = os.path.join(base_dir, "correct")
    offsets_json = os.path.join(base_dir, "slice_offsets.json") if "merge_error" in base_dir else os.path.join(base_dir, "slice_offsets.json")
    iou_threshold = 0.75
    output_json = args.output_json
    checkpoint_path = args.checkpoint
    device = args.device

    all_seg_files = [f for f in os.listdir(seg_dir) if f.endswith('.tiff')]
    if args.seg_filter == 'ms':
        seg_files = [f for f in all_seg_files if 'ms' in f]
    elif args.seg_filter == 'not_ms':
        seg_files = [f for f in all_seg_files if 'ms' not in f]
    else:
        seg_files = all_seg_files

    img_dict = {re.findall(r'\d+', f)[0]: f for f in os.listdir(img_dir) if f.endswith('.tiff') and re.findall(r'\d+', f)}
    correct_dict = {re.findall(r'\d+', f)[0]: f for f in os.listdir(correct_dir) if f.endswith('.tiff') and re.findall(r'\d+', f)}

    offsets = load_offsets(offsets_json)
    model = PointNet2Classification(num_classes=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    tasks = []
    for seg_file in seg_files:
        m = re.findall(r'\d+', seg_file)
        if not m:
            print(f"[SKIP] {seg_file}: no number in filename.")
            continue
        file_id = m[0]
        if file_id in img_dict and file_id in correct_dict:
            seg_path = os.path.join(seg_dir, seg_file)
            img_path = os.path.join(img_dir, img_dict[file_id])
            correct_path = os.path.join(correct_dir, correct_dict[file_id])
            tasks.append((seg_path, img_path, correct_path, offsets, model, device, iou_threshold))
        else:
            print(f"[SKIP] {file_id}: missing image or ground truth.")

    print(f"Processing {len(tasks)} files with {args.num_workers} workers...")
    results = []
    with Pool(processes=args.num_workers) as pool:
        for res in tqdm(pool.imap(worker, tasks), total=len(tasks)):
            if res:
                results.append(res)

    total_tp = sum(r["TP"] for r in results)
    total_fp = sum(r["FP"] for r in results)
    total_fn = sum(r["FN"] for r in results)
    global_precision, global_recall, global_f1 = compute_metrics(total_tp, total_fp, total_fn)
    avg_precision = np.mean([r["Precision"] for r in results])
    avg_recall = np.mean([r["Recall"] for r in results])
    avg_f1 = np.mean([r["F1"] for r in results])

    summary = {
        "per_image": results,
        "global_metrics": {
            "Precision": global_precision,
            "Recall": global_recall,
            "F1": global_f1
        },
        "average_metrics": {
            "Precision": avg_precision,
            "Recall": avg_recall,
            "F1": avg_f1
        }
    }

    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Results saved to {output_json}")

if __name__ == "__main__":
    main() 