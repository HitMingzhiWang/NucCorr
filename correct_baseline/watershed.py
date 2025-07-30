import numpy as np
from skimage.morphology import ball, erosion
from skimage.measure import label
from skimage.segmentation import watershed
import tifffile
import sys

def main(mask_tiff, img_tiff, output_tiff, erosion_radius=5):
    # 读取mask和原图
    mask = tifffile.imread(mask_tiff) > 0
    img = tifffile.imread(img_tiff)
    print(f"Mask shape: {mask.shape}, Img shape: {img.shape}")

    # 对mask做腐蚀
    selem = ball(erosion_radius)
    eroded = erosion(mask, selem)

    # 连通区域作为种子
    seeds = label(eroded)
    print(f"Number of seeds: {seeds.max()}")

    labels = watershed(img, seeds, mask=mask)

    tifffile.imwrite(output_tiff, labels.astype(np.uint16))
    print(f"Watershed result saved to {output_tiff}")

if __name__ == '__main__':

    main('/nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14/seg/8533661.tiff', '/nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14/img/8533661_img.tiff', '1.tiff',)