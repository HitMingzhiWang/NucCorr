import json

# 读取原始json
with open('/nvme2/mingzhi/NucCorr/correct_baseline/utils/resultsPrediction_merge.json', 'r') as f:
    data = json.load(f)

# 排序
sorted_per_image = sorted(data['per_image'], key=lambda x: x['F1'], reverse=True)

# 保存排序后的结果为新json
with open('/nvme2/mingzhi/NucCorr/correct_baseline/utils/resultsPrediction_split_sorted_by_f1.json', 'w') as f:
    json.dump({'per_image': sorted_per_image}, f, indent=2)

# 也可以打印前10名
print("Top 10 by F1:")
for item in sorted_per_image[:10]:
    print(f"{item['truth_file']} | F1: {item['F1']:.4f} | Precision: {item['Precision']:.4f} | Recall: {item['Recall']:.4f}")


