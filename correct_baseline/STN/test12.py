import os
import torch
import numpy as np
import tifffile
from pathlib import Path
import argparse
from model.stn import TETRIS_NucleiSegmentation
from dataset.dataset import mask_to_sdf

def crop_by_centroid(img, seg, crop_size=128):
    coords = np.argwhere(seg > 0)
    if coords.size == 0:
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

def save_results(seg, img, pred_sdf, gt_sdf, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    tifffile.imwrite(os.path.join(save_dir, 'input_seg.tiff'), seg.squeeze().cpu().numpy().astype(np.float32))
    tifffile.imwrite(os.path.join(save_dir, 'input_img.tiff'), img.squeeze().cpu().numpy().astype(np.float32))
    tifffile.imwrite(os.path.join(save_dir, 'pred_sdf.tiff'), pred_sdf.squeeze().cpu().numpy().astype(np.float32))
    tifffile.imwrite(os.path.join(save_dir, 'gt_sdf.tiff'), gt_sdf.squeeze().cpu().numpy().astype(np.float32))
    
    # SDF转mask并保存
    pred_mask = (pred_sdf.squeeze().cpu().numpy() < 0).astype(np.uint8)
    gt_mask = (gt_sdf.squeeze().cpu().numpy() < 0).astype(np.uint8)
    tifffile.imwrite(os.path.join(save_dir, 'pred_mask.tiff'), pred_mask)
    tifffile.imwrite(os.path.join(save_dir, 'gt_mask.tiff'), gt_mask)

def test(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 加载SDF模板
    sdf_template = np.load('/nvme2/mingzhi/NucCorr/correct_baseline/STN/nuclei_avg_sdf_template.npy')
    model = TETRIS_NucleiSegmentation(sdf_template=sdf_template).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    seg = tifffile.imread(args.seg_path)
    img = tifffile.imread(args.img_path)
    # 只保留前景
    seg[seg != args.fg_label] = 0
    seg[seg == args.fg_label] = 1
    seg = seg.astype(np.uint8)
    # crop
    crop_img, crop_seg = crop_by_centroid(img, seg, crop_size=128)
    # SDF GT
    gt_sdf = mask_to_sdf(crop_seg)
    # 转为tensor并加channel
    seg_tensor = torch.from_numpy(crop_seg).float().unsqueeze(0).unsqueeze(0)
    img_tensor = torch.from_numpy(crop_img).float().unsqueeze(0).unsqueeze(0) / 255.0
    gt_sdf_tensor = torch.from_numpy(gt_sdf).float().unsqueeze(0).unsqueeze(0)
    seg_tensor = seg_tensor.to(device)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        x = torch.cat([seg_tensor, img_tensor], dim=1)
        pred_sdf = model(x)
        save_results(seg_tensor, img_tensor, pred_sdf, gt_sdf_tensor, args.save_dir)
        print(f"Results saved to {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_path', type=str, default='/nvme2/mingzhi/selected/seg/595815.tiff', help='Path to input segmentation file')
    parser.add_argument('--img_path', type=str, default='/nvme2/mingzhi/selected/img/595815_img.tiff', help='Path to input image file')
    parser.add_argument('--model_path', type=str, default='/nvme2/mingzhi/NucCorr/correct_baseline/STN/checkpoints/best_model_epoch_0.pth', help='Path to trained model')
    parser.add_argument('--save_dir', type=str, default='/nvme2/mingzhi/selected/result', help='Directory to save results')
    parser.add_argument('--fg_label', type=int, default=21, help='Foreground label value in seg')
    args = parser.parse_args()
    test(args) 