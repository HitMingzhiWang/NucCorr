#!/bin/bash

python batch_eval_no_save.py \
  --base_dir /nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14 \
  --seg_filter ms \
  --output_json batch_eval_results_ms_midlnet.json \
  --checkpoint /nvme2/mingzhi/NucCorr/correct_baseline/MIDL/checkpoints/checkpoint_epoch_50.pth \
  --device cuda

python batch_eval_no_save.py \
  --base_dir /nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14 \
  --seg_filter not_ms \
  --output_json batch_eval_results_merge_midlnet.json \
  --checkpoint /nvme2/mingzhi/NucCorr/correct_baseline/MIDL/checkpoints/checkpoint_epoch_50.pth \
  --device cuda

python batch_eval_no_save.py \
  --base_dir /nvme2/mingzhi/NucCorr/NucCorrData/split_error \
  --seg_filter all \
  --output_json batch_eval_results_split_midlnet.json \
  --checkpoint /nvme2/mingzhi/NucCorr/correct_baseline/MIDL/checkpoint_epoch_50.pth \
  --device cuda