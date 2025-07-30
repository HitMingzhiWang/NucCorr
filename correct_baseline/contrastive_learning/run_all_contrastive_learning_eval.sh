#!/bin/bash

# 对比学习方法批量评估脚本

# 设置模型权重路径
CNN_CHECKPOINT="/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/3DCNN/logs/train_20250728-182654/checkpoint_best.pth"
POINTNET_CHECKPOINT="/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/pointNet/checkpoints/best_model.pth"
DEVICE="cuda"

echo "=== 开始对比学习方法批量评估 ==="
echo "3D CNN权重: $CNN_CHECKPOINT"
echo "PointNet2权重: $POINTNET_CHECKPOINT"
echo "设备: $DEVICE"
echo ""

# 1. 评估merge_error数据集的ms样本
echo "1. 评估merge_error数据集的ms样本..."
python batch_eval_no_save.py \
  --base_dir /nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14 \
  --seg_filter ms \
  --output_json batch_eval_results_ms_contrastive_learning.json \
  --cnn_checkpoint "$CNN_CHECKPOINT" \
  --pointnet_checkpoint "$POINTNET_CHECKPOINT" \
  --device "$DEVICE"

echo ""

# 2. 评估merge_error数据集的not_ms样本
echo "2. 评估merge_error数据集的not_ms样本..."
python batch_eval_no_save.py \
  --base_dir /nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14 \
  --seg_filter not_ms \
  --output_json batch_eval_results_merge_contrastive_learning.json \
  --cnn_checkpoint "$CNN_CHECKPOINT" \
  --pointnet_checkpoint "$POINTNET_CHECKPOINT" \
  --device "$DEVICE"

echo ""

# 3. 评估split_error数据集的所有样本
echo "3. 评估split_error数据集的所有样本..."
python batch_eval_no_save.py \
  --base_dir /nvme2/mingzhi/NucCorr/NucCorrData/split_error \
  --seg_filter all \
  --output_json batch_eval_results_split_contrastive_learning.json \
  --cnn_checkpoint "$CNN_CHECKPOINT" \
  --pointnet_checkpoint "$POINTNET_CHECKPOINT" \
  --device "$DEVICE"

echo ""
echo "=== 所有评估完成 ==="
echo "结果文件:"
echo "- batch_eval_results_ms_contrastive_learning.json"
echo "- batch_eval_results_merge_contrastive_learning.json"
echo "- batch_eval_results_split_contrastive_learning.json" 