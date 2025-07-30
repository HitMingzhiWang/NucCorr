import os
import numpy as np
import tifffile
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def count_ids_in_file(filepath):
    arr = tifffile.imread(filepath)
    unique_ids = np.unique(arr)
    unique_ids = unique_ids[unique_ids != 0]  # 排除背景
    return len(unique_ids), os.path.basename(filepath)

if __name__ == "__main__":
    folder = "/nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14/correct"  # 你的目标文件夹
    tiff_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.tif', '.tiff'))]
    total_ids = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(count_ids_in_file, tiff_files), total=len(tiff_files), desc="统计id数"))
    for num_ids, filename in results:
        print(f"{filename}: {num_ids} ids")
        total_ids += num_ids
    print(f"所有tiff文件中总id数: {total_ids}")