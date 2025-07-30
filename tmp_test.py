import numpy as np
import cc3d
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.morphology import ball
from tqdm import tqdm
from skimage import io

def detect_and_correct_merge_watershed(image, seg, iteration=7, erosion_shape="default", min_size=50, connectivity=6):
    """
    使用原图 image 引导 watershed，处理 false merge。
    """
    cc = cc3d.connected_components(seg, connectivity=connectivity)
    out = np.zeros_like(seg, dtype=np.int32)
    label_offset = 1

    for label in tqdm(range(1, cc.max() + 1)):
        region_mask = (cc == label)
        if np.count_nonzero(region_mask) < min_size:
            continue

        coords = np.argwhere(region_mask)
        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0) + 1

        region_crop = region_mask[zmin:zmax, ymin:ymax, xmin:xmax]
        image_crop = image[zmin:zmax, ymin:ymax, xmin:xmax]

        if erosion_shape != "default":
            structure = ball(radius=2)
            eroded = ndimage.binary_erosion(region_crop, structure=structure, iterations=iteration)
        else:
            eroded = ndimage.binary_erosion(region_crop, iterations=iteration)

        markers = cc3d.connected_components(eroded, connectivity=connectivity)

        if markers.max() == 0:
            markers = region_crop.astype(np.uint16)

        # 使用图像做分水岭 height map
        labels_ws = watershed(image_crop, markers=markers, mask=region_crop)

        for sub_id in range(1, labels_ws.max() + 1):
            mask_sub = (labels_ws == sub_id)
            if np.count_nonzero(mask_sub) < min_size:
                continue
            out[zmin:zmax, ymin:ymax, xmin:xmax][mask_sub] = label_offset
            label_offset += 1

    return out


# ==== 加载图像（转换为uint8）====
img = io.imread('/nvme2/mingzhi/NucCorr/1958996_aligned.tif').astype(np.uint8)
seg = io.imread('/nvme2/mingzhi/NucCorr/1958996_aligned_pred.tif').astype(np.uint8)

# ==== 执行分割 ====
out = detect_and_correct_merge_watershed(img, seg)

# ==== 检查最大label是否安全 ====
max_label = out.max()
print(max_label)
if max_label > 255:
    print(f"⚠ 警告: label数 {max_label} 超过 uint8 上限，保存为 uint8 会导致错误！")
    # 可选择保存为 uint16
    io.imsave('/nvme2/mingzhi/NucCorr/1958996_out_uint16.tiff', out.astype(np.uint16))
else:
    io.imsave('/nvme2/mingzhi/NucCorr/1958996_out.tiff', out.astype(np.uint8))
