python batch_process_zernike.py \
  --base_dir /nvme2/mingzhi/NucCorr/NucCorrData/split_error \
  --output_dir /nvme2/mingzhi/NucCorr/correct_baseline/split_test_results/0.5 \
  --model_path /nvme2/mingzhi/NucCorr/complete_model/checkpoints/best_model_epoch_bce.pth \
  --gmm_model_path /nvme2/mingzhi/NucCorr/NucDet/gmm_zernike_model.pkl

python batch_process_zernike.py \
  --base_dir /nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14 \
  --output_dir /nvme2/mingzhi/NucCorr/correct_baseline/merge_test_results/0.5 \
  --model_path /nvme2/mingzhi/NucCorr/complete_model/checkpoints/best_model_epoch_bce.pth \
  --gmm_model_path /nvme2/mingzhi/NucCorr/NucDet/gmm_zernike_model.pkl