import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.io import imread
from skimage.measure import label
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 添加一个线程锁用于打印
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

def read_tif(file_path):
    """读取二值化的实例分割 TIF 文件"""
    img = imread(file_path)
    return img.astype(np.int32)

def compute_iou(pred_region, truth_region):
    """计算两个区域之间的 IoU"""
    intersection = np.sum(np.logical_and(pred_region, truth_region))
    union = np.sum(np.logical_or(pred_region, truth_region))
    return intersection / union if union != 0 else 0

def match_instances(truth_labels, pred_labels, iou_threshold=0.8):
    """
    使用匈牙利算法匹配真实和预测实例
    返回: tp, fp, fn
    """
    truth_ids = np.unique(truth_labels)[1:]
    pred_ids = np.unique(pred_labels)[1:]

    
    if len(truth_ids) == 0 and len(pred_ids) == 0:
        return 0, 0, 0  
    elif len(truth_ids) == 0:
        return 0, len(pred_ids), 0  
    elif len(pred_ids) == 0:
        return 0, 0, len(truth_ids)  

 
    iou_matrix = np.zeros((len(truth_ids), len(pred_ids)))
    for i, tid in enumerate(truth_ids):
        truth_mask = (truth_labels == tid)
        for j, pid in enumerate(pred_ids):
            pred_mask = (pred_labels == pid)
            iou_matrix[i, j] = compute_iou(pred_mask, truth_mask)

  
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)  

    
    tp = 0
    matched_truth = set()
    matched_pred = set()
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            tp += 1
            matched_truth.add(truth_ids[r])
            matched_pred.add(pred_ids[c])

  
    fp = len(pred_ids) - len(matched_pred)  
    fn = len(truth_ids) - len(matched_truth) 

    return tp, fp, fn

def compute_metrics(tp, fp, fn):
    """计算 Precision, Recall, F1"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def evaluate_image_pair(truth_path, pred_path, iou_threshold=0.8):
    """评估单对真值-预测图像"""
   
    truth_img = read_tif(truth_path)
    pred_img = read_tif(pred_path)
    truth_labels = truth_img.astype(np.int32)
    pred_labels = pred_img.astype(np.int32)

    
    tp, fp, fn = match_instances(truth_labels, pred_labels, iou_threshold)
    precision, recall, f1 = compute_metrics(tp, fp, fn)

    # 添加文件名信息
    return {
        "truth_file": os.path.basename(truth_path),
        "pred_file": os.path.basename(pred_path),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

def find_matching_pairs(truth_dir, pred_dir):
    # 只获取包含 'mis' 的真值文件
    truth_files = sorted([f for f in os.listdir(truth_dir) if f.endswith('.tiff')])
    pred_files = sorted(os.listdir(pred_dir))
    
    # 预处理所有预测文件的数字序列
    pred_numbers_dict = {}
    for pfile in pred_files:
        numbers = tuple(re.findall(r'\d+', pfile))
        if numbers:
            pred_numbers_dict[numbers] = pfile
    
    matched_pairs = []
    for tfile in truth_files:
        tfile_numbers = tuple(re.findall(r'\d+', tfile))
        if not tfile_numbers:
            safe_print(f"警告: {tfile} 中没有找到数字")
            continue
            
        if tfile_numbers in pred_numbers_dict:
            pfile = pred_numbers_dict[tfile_numbers]
            matched_pairs.append(
                (os.path.join(truth_dir, tfile), os.path.join(pred_dir, pfile))
            )
        else:
            safe_print(f"警告: 未找到与 {tfile} 匹配的预测文件")
    
    return matched_pairs

def process_image_pair(args):
    truth_path, pred_path, iou_threshold = args
    try:
        metrics = evaluate_image_pair(truth_path, pred_path, iou_threshold)
        safe_print(f"已完成: {os.path.basename(truth_path)} vs {os.path.basename(pred_path)}")
        return metrics
    except Exception as e:
        safe_print(f"处理 {os.path.basename(truth_path)} 时出错: {str(e)}")
        return None

def evaluate_all(truth_dir, pred_dir, iou_threshold=0.8, max_workers=8):
    matched_pairs = find_matching_pairs(truth_dir, pred_dir)
    if not matched_pairs:
        raise ValueError("未找到匹配的真值-预测文件对")

    all_results = []
    total_tp, total_fp, total_fn = 0, 0, 0
    
    # 准备参数列表
    args_list = [(truth_path, pred_path, iou_threshold) for truth_path, pred_path in matched_pairs]
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair = {executor.submit(process_image_pair, args): args for args in args_list}
        
        for future in as_completed(future_to_pair):
            metrics = future.result()
            if metrics is not None:
                all_results.append(metrics)
                total_tp += metrics["TP"]
                total_fp += metrics["FP"]
                total_fn += metrics["FN"]

    global_precision, global_recall, global_f1 = compute_metrics(total_tp, total_fp, total_fn)
    
    avg_precision = np.mean([r["Precision"] for r in all_results])
    avg_recall = np.mean([r["Recall"] for r in all_results])
    avg_f1 = np.mean([r["F1"] for r in all_results])

    return {
        "per_image": all_results,
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

if __name__ == "__main__":
    TRUTH_DIR = "/nvme2/mingzhi/NucCorr/NucCorrData/split_error/correct"  
    PRED_DIR = "/nvme2/mingzhi/NucCorr/correct_baseline/split_test_results/pred_larger"                         
    IOU_THRESHOLD = 0.75                              
    MAX_WORKERS = 32  # 可以根据CPU核心数调整

    results = evaluate_all(TRUTH_DIR, PRED_DIR, IOU_THRESHOLD, MAX_WORKERS)

    # 修改输出信息，明确说明只统计了包含 'mis' 的文件
    print("\n只统计文件名包含 'mis' 的文件的结果:")
    print("\n逐文件结果:")
    for i, res in enumerate(results["per_image"]):
        print(f"图像 {i+1}:")
        print(f"  文件名: {res['truth_file']}")
        print(f"  TP: {res['TP']}, FP: {res['FP']}, FN: {res['FN']}")
        print(f"  Precision: {res['Precision']:.4f}")
        print(f"  Recall:    {res['Recall']:.4f}")
        print(f"  F1:        {res['F1']:.4f}")

    print("\n全局指标 (基于总TP/FP/FN):")
    gm = results["global_metrics"]
    print(f"Precision: {gm['Precision']:.4f}")
    print(f"Recall:    {gm['Recall']:.4f}")
    print(f"F1:        {gm['F1']:.4f}")

    print("\n平均指标 (各文件指标的平均值):")
    am = results["average_metrics"]
    print(f"Precision: {am['Precision']:.4f}")
    print(f"Recall:    {am['Recall']:.4f}")
    print(f"F1:        {am['F1']:.4f}")

    # 将结果保存到JSON文件
    output_json_path = "/nvme2/mingzhi/NucCorr/correct_baseline/utils/resultsPrediction_split.json"  
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n评估结果已保存到 {output_json_path}")


