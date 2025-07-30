import json

files = [
    "/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/batch_eval_results_merge_contrastive_learning.json",
    "/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/batch_eval_results_ms_contrastive_learning.json",
    "/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/batch_eval_results_split_contrastive_learning.json",
]

all_prec, all_rec, all_f1 = [], [], []
total_tp, total_fp, total_fn = 0, 0, 0
has_tp_fp_fn = False

for file in files:
    with open(file) as f:
        data = json.load(f)
    if 'per_image' in data:
        items = data['per_image']
    elif isinstance(data, dict):
        items = data.values()
    else:
        items = data
    for item in items:
        # 统计TP/FP/FN
        if all(k in item for k in ['TP', 'FP', 'FN']):
            has_tp_fp_fn = True
            total_tp += item['TP']
            total_fp += item['FP']
            total_fn += item['FN']
        # 统计Precision/Recall/F1
        if 'Precision' in item and 'Recall' in item and 'F1' in item:
            all_prec.append(item['Precision'])
            all_rec.append(item['Recall'])
            all_f1.append(item['F1'])

if has_tp_fp_fn:
    # micro平均
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"MICRO PRECISION: {precision:.4f}, RECALL: {recall:.4f}, F1: {f1:.4f}")
else:
    # macro平均
    precision = sum(all_prec) / len(all_prec) if all_prec else 0
    recall = sum(all_rec) / len(all_rec) if all_rec else 0
    f1 = sum(all_f1) / len(all_f1) if all_f1 else 0
    print(f"MACRO PRECISION: {precision:.4f}, RECALL: {recall:.4f}, F1: {f1:.4f}")