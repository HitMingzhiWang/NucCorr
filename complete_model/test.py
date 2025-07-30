import os
import torch
import numpy as np
import tifffile
from pathlib import Path
import argparse
from model.unet import get_model
import matplotlib.pyplot as plt

def crop_by_centroid(img, seg, crop_size=128):
    # seg: numpy array, img: numpy array, both shape (D, H, W)
    coords = np.argwhere(seg > 0)
    if coords.size == 0:
        # 没有前景，直接中心crop
        cz, cy, cx = [s // 2 for s in seg.shape]
    else:
        cz, cy, cx = np.round(coords.mean(axis=0)).astype(int)
    sz = crop_size
    D, H, W = seg.shape
    startz = cz - sz // 2
    starty = cy - sz // 2
    startx = cx - sz // 2
    endz = startz + sz
    endy = starty + sz
    endx = startx + sz
    crop_seg = np.zeros((sz, sz, sz), dtype=seg.dtype)
    crop_img = np.zeros((sz, sz, sz), dtype=img.dtype)
    # 计算原图和crop的重叠区域
    src_z1 = max(0, startz)
    src_y1 = max(0, starty)
    src_x1 = max(0, startx)
    src_z2 = min(D, endz)
    src_y2 = min(H, endy)
    src_x2 = min(W, endx)
    dst_z1 = src_z1 - startz
    dst_y1 = src_y1 - starty
    dst_x1 = src_x1 - startx
    dst_z2 = dst_z1 + (src_z2 - src_z1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    crop_seg[dst_z1:dst_z2, dst_y1:dst_y2, dst_x1:dst_x2] = seg[src_z1:src_z2, src_y1:src_y2, src_x1:src_x2]
    crop_img[dst_z1:dst_z2, dst_y1:dst_y2, dst_x1:dst_x2] = img[src_z1:src_z2, src_y1:src_y2, src_x1:src_x2]
    return crop_img, crop_seg

def save_results(seg, img, pred, score, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    tifffile.imwrite(os.path.join(save_dir, 'input_seg.tiff'), seg.squeeze().cpu().numpy().astype(np.float32))
    tifffile.imwrite(os.path.join(save_dir, 'input_img.tiff'), img.squeeze().cpu().numpy().astype(np.float32))
    tifffile.imwrite(os.path.join(save_dir, 'pred.tiff'), pred.squeeze().cpu().numpy().astype(np.float32))
    tifffile.imwrite(os.path.join(save_dir, 'score.tiff'), score.squeeze().cpu().numpy().astype(np.float32))

def save_overlay_heatmap_slices(img, score, save_dir, prefix='overlay'):
    out_dir = os.path.join(save_dir, 'overlay_heatmap')
    os.makedirs(out_dir, exist_ok=True)
    img_np = img.squeeze().cpu().numpy()
    score_np = score.squeeze().cpu().numpy()
    # 归一化
    img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    score_norm = (score_np - score_np.min()) / (score_np.max() - score_np.min() + 1e-8)
    for z in range(img_norm.shape[0]):
        img_slice = img_norm[z]
        score_slice = score_norm[z]
        # 灰度转RGB
        img_rgb = np.stack([img_slice]*3, axis=-1)
        # 生成heatmap
        cmap = plt.get_cmap('jet')
        heatmap = cmap(score_slice)[..., :3]  # (H, W, 3), float
        # 叠加
        alpha = 0.5
        overlay = (1 - alpha) * img_rgb + alpha * heatmap
        overlay = np.clip(overlay, 0, 1)
        # 保存
        plt.imsave(
            os.path.join(out_dir, f'{prefix}_z{z:03d}.png'),
            overlay
        )

def test(args):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = get_model().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    seg = tifffile.imread(args.seg_path)
    img = tifffile.imread(args.img_path)
    seg[seg!=19] = 0
    seg[seg==19] = 1
    seg = seg.astype(np.uint8)
    # crop
    crop_img, crop_seg = crop_by_centroid(img, seg, crop_size=128)
    # 转换为tensor并添加channel维度
    seg_tensor = torch.from_numpy(crop_seg).float().unsqueeze(0).unsqueeze(0)
    img_tensor = torch.from_numpy(crop_img).float().unsqueeze(0).unsqueeze(0) / 255.0
    seg_tensor = seg_tensor.to(device)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        x = torch.cat([seg_tensor, img_tensor], dim=1)
        output = model(x)
        score = torch.sigmoid(output)
        pred = (score > 0.5).float()
        save_results(seg_tensor, img_tensor, pred, score, args.save_dir)
        save_overlay_heatmap_slices(img_tensor, score, args.save_dir)
        print(f"Results saved to {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_path', type=str, default='/nvme2/mingzhi/selected/seg/4389032.tiff', help='Path to input segmentation file')
    parser.add_argument('--img_path', type=str, default='/nvme2/mingzhi/selected/img/4389032_img.tiff', help='Path to input image file')
    parser.add_argument('--model_path', type=str, default='/nvme2/mingzhi/NucCorr/complete_model/checkpoints/best_model_epoch_32.pth', help='Path to trained model')
    parser.add_argument('--save_dir', type=str, default='/nvme2/mingzhi/selected/result', help='Directory to save results')
    args = parser.parse_args()
    test(args)