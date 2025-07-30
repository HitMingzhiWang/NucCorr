import os
import argparse
from utils.error_helper import *
from baseline_zerniker import merge_error_correction, split_error_correction, restore_original_segmentation, load_zernike_cache
from skimage import io
import numpy as np
from utils.apply_offsets import load_offsets
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import torch
import joblib

def process_single_file(args):
    seg_path, img_path, offsets, output_dir, device, model_path, zernike_cache, gmm_model = args
    try:
        img = io.imread(img_path)
        seg = io.imread(seg_path)
        file_id = os.path.basename(seg_path).split('.')[0]
        print(f"\nProcessing file: {file_id} on {device}")

        if str(file_id) in offsets:
            aligned_img, corrected_seg, shifts, original_shape = merge_error_correction(img, seg, file_id, offsets, output_dir)
        else:
            aligned_img = img
            corrected_seg = process_all_components_with_safe_merge(seg, iteration=2, erosion_shape='ball')
            shifts = None
            original_shape = img.shape

        print("Step 5: Performing split error correction...")
        final_seg = split_error_correction(aligned_img, corrected_seg, model_path, gmm_model, zernike_cache, device=device)

        if shifts is not None:
            print("Step 6: Restoring segmentation to original space...")
            restored_seg = restore_original_segmentation(final_seg, shifts, original_shape)
        else:
            restored_seg = final_seg

        final_output_dir = os.path.join(output_dir, 'pred')
        os.makedirs(final_output_dir, exist_ok=True)
        final_seg_path = os.path.join(final_output_dir, f'{file_id}_pred.tiff')
        io.imsave(final_seg_path, restored_seg.astype(np.uint8))
        print(f"Successfully processed {file_id}")
        return True
    except Exception as e:
        print(f"Error processing {file_id}: {str(e)}")
        return False

def worker_init(device, gmm_model_path):
    global _zernike_cache, _gmm_model
    _zernike_cache = load_zernike_cache(max_order=20, device=device)
    _gmm_model = joblib.load(gmm_model_path)

def process_single_file_with_cache(args):
    seg_path, img_path, offsets, output_dir, device, model_path = args
    global _zernike_cache, _gmm_model
    return process_single_file((seg_path, img_path, offsets, output_dir, device, model_path, _zernike_cache, _gmm_model))

def batch_process(base_dir, output_dir, model_path, gmm_model_path):
    seg_dir = os.path.join(base_dir, "seg")
    img_dir = os.path.join(base_dir, "img")
    offsets_json = os.path.join(base_dir, "slice_offsets.json")

    offsets = load_offsets(offsets_json)
    seg_files = [f for f in os.listdir(seg_dir) if f.endswith('.tiff')]
    total_files = len(seg_files)
    print(f"Found {total_files} files to process in {base_dir}")

    # 固定只用一张GPU
    if torch.cuda.is_available():
        device = "cuda:2"
    else:
        device = "cpu"
    devices = [device]

    tasks = []
    for seg_file in seg_files:
        file_id = seg_file.split('.')[0]
        seg_path = os.path.join(seg_dir, seg_file)
        img_path = os.path.join(img_dir, f"{file_id}_img.tiff")
        tasks.append((seg_path, img_path, offsets, output_dir, device, model_path))

    successful = 0
    with ProcessPoolExecutor(max_workers=8, initializer=worker_init, initargs=(device, gmm_model_path)) as executor:
        future_to_task = {executor.submit(process_single_file_with_cache, task): task for task in tasks}
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc=f"Processing files in {base_dir}"):
            if future.result():
                successful += 1

    print(f"\nProcessing completed for {base_dir}!")
    print(f"Successfully processed: {successful}/{total_files} files")

def main():
    parser = argparse.ArgumentParser(description="Batch process zernike split/merge error correction.")
    parser.add_argument('--base_dir', required=True, help='Base directory path')
    parser.add_argument('--output_dir', required=True, help='Output directory path')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--gmm_model_path', type=str, required=True, help='Path to GMM model')
    args = parser.parse_args()

    batch_process(args.base_dir, args.output_dir, args.model_path, args.gmm_model_path)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main() 