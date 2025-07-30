#!/bin/bash
python batch_process.py \
  --base_dir /nvme2/mingzhi/NucCorr/NucCorrData/split_error \
  --model_path /nvme2/mingzhi/NucCorr/complete_model/checkpoints/best_model_epoch_bce.pth \
  --output_dir /nvme2/mingzhi/NucCorr/correct_baseline/split_test_results

python batch_process.py \
  --base_dir /nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14 \
  --model_path /nvme2/mingzhi/NucCorr/complete_model/checkpoints/best_model_epoch_bce.pth \
  --output_dir /nvme2/mingzhi/NucCorr/correct_baseline/merge_test_results