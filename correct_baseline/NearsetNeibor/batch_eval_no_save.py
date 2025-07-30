import os
import re
import numpy as np
from tifffile import imread as tiff_imread
from tqdm import tqdm
import sys
import json
from multiprocessing import Pool, cpu_count

# 添加模块路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/nvme2/mingzhi/NucCorr')

from correct_baseline.utils.apply_offsets import load_offsets, restore_original_segmentation
from correct_baseline.utils.error_helper import process_all_components_with_safe_merge
from correct_baseline.NearsetNeibor.baseline import merge_error_correction, split_error_correction
from correct_baseline.utils.evaluate import match_instances, compute_metrics


def evaluate_pair(truth, pred, iou_threshold=0.8):
    truth_labels = truth.astype(np.int32)
    pred_labels = pred.astype(np.int32)
    tp, fp, fn = match_instances(truth_labels, pred_labels, iou_threshold)
    precision, recall, f1 = compute_metrics(tp, fp, fn)
    return {
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": precision, "Recall": recall, "F1": f1
    }


def process_and_eval(seg_path, img_path, correct_path, offsets, iou_threshold=0.8):
    file_id = re.findall(r'\d+', os.path.basename(seg_path))[0]

    img = tiff_imread(img_path)
    seg = tiff_imread(seg_path)
    correct = tiff_imread(correct_path)

    aligned_img, corrected_seg, shifts, original_shape, id1_related_ids = merge_error_correction(img, seg, file_id, offsets, None)

    final_seg = split_error_correction(corrected_seg, id1_related_ids)

    if shifts is not None:
        restored_seg = restore_original_segmentation(final_seg, shifts, original_shape)
    else:
        restored_seg = final_seg

    metrics = evaluate_pair(correct, restored_seg, iou_threshold)
    metrics["file_id"] = file_id
    return metrics


# 用于多进程
def worker(task):
    try:
        seg_path, img_path, correct_path, offsets, iou_threshold = task
        if not os.path.exists(img_path) or not os.path.exists(correct_path):
            return None
        return process_and_eval(seg_path, img_path, correct_path, offsets, iou_threshold)
    except Exception as e:
        print(f"[ERROR] {seg_path}: {e}")
        return None


def main():
    base_dir = "/nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14"
    seg_dir = os.path.join(base_dir, "match_seg")
    img_dir = os.path.join(base_dir, "img")
    correct_dir = os.path.join(base_dir, "correct")
    offsets_json = os.path.join(base_dir, "slice_offsets_mid_dx.json")
    iou_threshold = 0.75
    output_json = "batch_eval_results_merge.json"

    offsets = load_offsets(offsets_json)
    seg_files = [f for f in os.listdir(seg_dir) if f.endswith('.tiff') and 'ms' in f ]

    # Step 1: 缓存 img 和 correct 的文件映射
    img_dict = {re.findall(r'\d+', f)[0]: f for f in os.listdir(img_dir) if f.endswith('.tiff')}
    correct_dict = {re.findall(r'\d+', f)[0]: f for f in os.listdir(correct_dir) if f.endswith('.tiff')}

    # Step 2: 构建任务列表
    tasks = []
    for seg_file in seg_files:
        file_id = re.findall(r'\d+', seg_file)[0]
        if file_id in img_dict and file_id in correct_dict:
            seg_path = os.path.join(seg_dir, seg_file)
            img_path = os.path.join(img_dir, img_dict[file_id])
            correct_path = os.path.join(correct_dir, correct_dict[file_id])
            tasks.append((seg_path, img_path, correct_path, offsets, iou_threshold))
        else:
            print(f"[SKIP] {file_id}: missing image or ground truth.")

    # Step 3: 多进程执行任务
    print(f"Processing {len(tasks)} files with {cpu_count() // 2} workers...")
    results = []
    with Pool(processes=cpu_count() // 2) as pool:
        for res in tqdm(pool.imap(worker, tasks), total=len(tasks)):
            if res:
                results.append(res)

    # Step 4: 汇总指标
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
