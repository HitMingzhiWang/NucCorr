import numpy as np
import tifffile

# 读取npy文件
sdf = np.load('/nvme2/mingzhi/NucCorr/correct_baseline/STN/nuclei_avg_sdf_template.npy')

# 直接保存为float32 tiff
tifffile.imwrite('nuclei_avg_sdf_template.tiff', sdf.astype(np.float32))

# 可选：归一化到0-255保存为uint8，便于ImageJ等软件伪彩色显示
sdf_norm = (sdf - sdf.min()) / (sdf.max() - sdf.min())
tifffile.imwrite('nuclei_avg_sdf_template_uint8.tiff', (sdf_norm).astype(np.float32))