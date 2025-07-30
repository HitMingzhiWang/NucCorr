import os
from utils.error_helper import *
from baseline import merge_error_correction, split_error_correction, restore_original_segmentation
from skimage import io
import numpy as np
from utils.apply_offsets import load_offsets
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torch
import argparse

def process_single_file(seg_path, img_path, offsets, output_dir, device, model_path):
    """处理单个文件"""
    try:
        # 读取数据
        img = io.imread(img_path)
        seg = io.imread(seg_path)
        file_id = os.path.basename(seg_path).split('.')[0]
        
        print(f"\nProcessing file: {file_id}")
        
        # 检查是否已经存在预测结果
        pred_path = os.path.join(output_dir, 'final_results', f'{file_id}_pred.tiff')
        #if os.path.exists(pred_path):
            #print(f"Prediction already exists for {file_id}, skipping...")
            #return True
        
        if str(file_id) in offsets:
            print("Performing alignment...")
            aligned_img, corrected_seg, shifts, original_shape = merge_error_correction(img, seg, file_id, offsets, output_dir)
        else:
            print("No alignment needed, using original data...")
            aligned_img = img
            corrected_seg = process_all_components_with_safe_merge(seg,iteration=2,erosion_shape='ball')
            shifts = None
            original_shape = img.shape
        
        # 执行split error correction
        print("Step 5: Performing split error correction...")
        final_seg = split_error_correction(aligned_img, corrected_seg, model_path, device)
        
        # 执行restore操作（如果需要）
        if shifts is not None:
            print("Step 6: Restoring segmentation to original space...")
            restored_seg = restore_original_segmentation(final_seg, shifts, original_shape)
        else:
            restored_seg = final_seg
        
        # 保存结果
        final_output_dir = os.path.join(output_dir, 'pred_larger')
        os.makedirs(final_output_dir, exist_ok=True)
        
        final_seg_path = os.path.join(final_output_dir, f'{file_id}_pred.tiff')
        io.imsave(final_seg_path, restored_seg.astype(np.uint8))
        
        print(f"Successfully processed {file_id}")
        return True
        
    except Exception as e:
        print(f"Error processing {file_id}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch process segmentation correction.")
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing seg, img, and slice_offsets.json')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of parallel workers (default: number of GPUs or 1)')
    args = parser.parse_args()

    base_dir = args.base_dir
    seg_dir = os.path.join(base_dir, "seg")
    img_dir = os.path.join(base_dir, "img")
    offsets_json = os.path.join(base_dir, "slice_offsets.json")
    model_path = args.model_path
    output_dir = args.output_dir

    # 加载offsets
    offsets = load_offsets(offsets_json)

    # 获取所有seg文件
    seg_files = [f for f in os.listdir(seg_dir) if f.endswith('.tiff')]
    total_files = len(seg_files)

    print(f"Found {total_files} files to process")

    # 设置GPU设备
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        devices = ["cpu"]
    else:
        devices = [f"cuda:{i}" for i in range(num_gpus)]

    # 允许用户指定最大并发数
    if args.max_workers is not None:
        max_workers = args.max_workers
    else:
        max_workers = len(devices)

    # 创建任务列表
    tasks = []
    for i, seg_file in enumerate(seg_files):
        file_id = seg_file.split('.')[0]
        seg_path = os.path.join(seg_dir, seg_file)
        img_path = os.path.join(img_dir, f"{file_id}_img.tiff")
        device = devices[i % len(devices)]  # 循环分配GPU
        tasks.append((seg_path, img_path, offsets, output_dir, device, model_path))

    # 使用线程池处理文件
    successful = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(process_single_file, *task): task 
            for task in tasks
        }

        # 使用tqdm显示进度
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Processing files"):
            if future.result():
                successful += 1

    print(f"\nProcessing completed!")
    print(f"Successfully processed: {successful}/{total_files} files")

if __name__ == "__main__":
    main() 