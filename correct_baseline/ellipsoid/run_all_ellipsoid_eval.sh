#!/bin/bash

python batch_eval_no_save.py \
  --base_dir /nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14 \
  --seg_filter ms \
  --output_json batch_eval_results_ms_ellipsoid.json

python batch_eval_no_save.py \
  --base_dir /nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14 \
  --seg_filter not_ms \
  --output_json batch_eval_results_merge_ellipsoid.json

python batch_eval_no_save.py \
  --base_dir /nvme2/mingzhi/NucCorr/NucCorrData/split_error \
  --seg_filter all \
  --output_json batch_eval_results_split_ellipsoid.json