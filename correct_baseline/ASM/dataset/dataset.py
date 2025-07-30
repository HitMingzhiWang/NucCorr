import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import sys
sys.path.append('/nvme2/mingzhi/NucCorr')
from torchvision import transforms
import tifffile
from scipy.ndimage import shift
from correct_baseline.utils.error_helper import *

class NucCorrDataset(Dataset):
    def __init__(self, root_dir, split_file, transform=None, img_size=(128, 128, 128), train=True, seed=42, points_npz_path=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.train = train
        self.seed = seed
        self.points_dict = None
        if points_npz_path is not None:
            self.points_dict = dict(np.load(points_npz_path, allow_pickle=True))
        self.rng = np.random.RandomState(seed)
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(seed)
        with open(split_file, 'r') as f:
            self.file_list = [line.strip() + '.tiff' for line in f.readlines()]
        self.img_files = []
        self.seg_files = []
        for filename in self.file_list:
            img_path = os.path.join(root_dir, 'img', filename)
            seg_path = os.path.join(root_dir, 'seg', filename)
            if os.path.exists(img_path) and os.path.exists(seg_path):
                self.img_files.append(filename)
                self.seg_files.append(filename)
            else:
                print(f"Warning: Missing files for {filename}")

    def _get_random_state(self):
        return {
            'numpy': self.rng.get_state(),
            'torch': self.torch_rng.get_state()
        }
    def _set_random_state(self, state):
        self.rng.set_state(state['numpy'])
        self.torch_rng.set_state(state['torch'])
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = os.path.join(self.root_dir, 'img', self.img_files[idx])
        seg_path = os.path.join(self.root_dir, 'seg', self.seg_files[idx])
        image = tifffile.imread(img_path)
        segmentation = tifffile.imread(seg_path)
        assert image.shape == self.img_size, f"Image shape {image.shape} doesn't match expected shape {self.img_size}"
        assert segmentation.shape == self.img_size, f"Segmentation shape {segmentation.shape} doesn't match expected shape {self.img_size}"
        segmentation[segmentation > 0] = 1
        image = torch.from_numpy(image).float()
        segmentation = torch.from_numpy(segmentation).long()
        image = image.unsqueeze(0)
        segmentation = segmentation.unsqueeze(0)
        image = image / 255.0
        complete_mask = segmentation.clone()
        last_shift = None
        if True:
            segmentation, dropped_regions, drop_type, dropped_blocks = self._apply_seg_augmentations(segmentation)
            image = self._apply_img_augmentations(image, segmentation, dropped_regions, drop_type, dropped_blocks)
            if self.transform and random.random() < 1:
                if isinstance(self.transform, transforms.Compose):
                    for t in self.transform.transforms:
                        if isinstance(t, RandomShift3DWrapper):
                            t.set_segmentation(segmentation)
                            t.set_complete_mask(complete_mask)
                image = self.transform(image)
                for t in self.transform.transforms:
                    if isinstance(t, RandomShift3DWrapper):
                        segmentation = t.get_shifted_segmentation()
                        complete_mask = t.get_shifted_complete_mask()
                        if hasattr(t.shift_transform, 'shifts') and t.shift_transform.shifts is not None:
                            last_shift = [int(s) for s in t.shift_transform.shifts]
                        else:
                            last_shift = None
            else:
                last_shift = None
        file_id = os.path.splitext(self.img_files[idx])[0]
        key = f"{file_id}_orig"
        complete_points = self.points_dict[key]
        if last_shift is not None:
            complete_points = complete_points + np.array(last_shift)
        complete_points = torch.from_numpy(complete_points).float()
        return segmentation, image, complete_mask, complete_points
    def _apply_seg_augmentations(self, seg):
        dropped_regions = torch.zeros_like(seg.squeeze(0))
        drop_type = random.choice([-1, 0, 1, 2, 3])
        dropped_blocks = []
        if self.rng.random() < 0.9:
            foreground = seg.squeeze(0).bool()
            if not foreground.any():
                return seg, dropped_regions, drop_type, dropped_blocks
            drop_type = self.rng.randint(0, 4)
            if drop_type == 0:
                block_size = (8, 8, 8)
                max_dropout_ratio = 0.9
                d, h, w = foreground.shape
                if d >= block_size[0] and h >= block_size[1] and w >= block_size[2]:
                    pool = torch.nn.MaxPool3d(
                        kernel_size=block_size,
                        stride=block_size,
                        padding=0
                    )
                    foreground_4d = foreground.unsqueeze(0).unsqueeze(0).float()
                    pooled = pool(foreground_4d).squeeze()
                    candidate_blocks = torch.nonzero(pooled > 0, as_tuple=False)
                    num_candidates = candidate_blocks.size(0)
                    if num_candidates > 0:
                        dropout_ratio = self.rng.uniform(0, max_dropout_ratio)
                        num_drop = int(num_candidates * dropout_ratio)
                        num_drop = max(0, min(num_drop, num_candidates))
                        if num_drop > 0:
                            drop_indices = torch.randperm(num_candidates, generator=self.torch_rng)[:num_drop]
                            blocks_to_drop = candidate_blocks[drop_indices]
                            for block in blocks_to_drop:
                                dz, dy, dx = block.tolist()
                                z_start = int(dz * block_size[0])
                                y_start = int(dy * block_size[1])
                                x_start = int(dx * block_size[2])
                                z_end = int(min(z_start + block_size[0], d))
                                y_end = int(min(y_start + block_size[1], h))
                                x_end = int(min(x_start + block_size[2], w))
                                if z_start < z_end and y_start < y_end and x_start < x_end:
                                    seg[0, z_start:z_end, y_start:y_end, x_start:x_end] = 0
                                    dropped_regions[z_start:z_end, y_start:y_end, x_start:x_end] = 1
                                    dropped_blocks.append((z_start, z_end, y_start, y_end, x_start, x_end))
            else:
                z_coords = torch.any(foreground, dim=(1, 2))
                nonzero_z = torch.nonzero(z_coords, as_tuple=True)[0]
                if nonzero_z.numel() > 0:
                    z_start = int(nonzero_z.min().item())
                    z_end = int(nonzero_z.max().item())
                    mask_z_length = z_end - z_start + 1
                    max_mask_length = int(mask_z_length * 0.9)
                    if max_mask_length > 0:
                        if drop_type == 1:
                            mask_length = self.rng.randint(0, max_mask_length)
                            m_start = z_start
                            m_end = int(min(z_start + mask_length, z_end + 1))
                            if m_start < m_end:
                                seg[0, m_start:m_end, :, :] = 0
                                dropped_regions[m_start:m_end, :, :] = 1
                        elif drop_type == 2:
                            mid_point = int((z_start + z_end) // 2)
                            mask_length = self.rng.randint(0, max_mask_length // 2)
                            m_start = int(max(z_start, mid_point - mask_length))
                            m_end = int(min(z_end + 1, mid_point + mask_length))
                            if m_start < m_end:
                                seg[0, m_start:m_end, :, :] = 0
                                dropped_regions[m_start:m_end, :, :] = 1
                        else:
                            mask_length = self.rng.randint(0, max_mask_length)
                            m_start = int(max(z_start, z_end - mask_length + 1))
                            m_end = z_end + 1
                            if m_start < m_end:
                                seg[0, m_start:m_end, :, :] = 0
                                dropped_regions[m_start:m_end, :, :] = 1
        return seg, dropped_regions, drop_type, dropped_blocks
    def _apply_img_augmentations(self, img, seg, dropped_regions, drop_type, dropped_blocks):
        if self.rng.random() < 1:
            d, h, w = dropped_regions.shape
            if drop_type == 0:
                num_blocks = len(dropped_blocks)
                if num_blocks > 0:
                    num_drop = max(1, int(num_blocks * 0.9))
                    selected_indices = torch.randperm(num_blocks, generator=self.torch_rng)[:num_drop]
                    for idx in selected_indices:
                        z_start, z_end, y_start, y_end, x_start, x_end = dropped_blocks[idx]
                        img[0, z_start:z_end, y_start:y_end, x_start:x_end] = 0
            else:
                dropped_slices = torch.any(dropped_regions, dim=(1, 2))
                dropped_slice_indices = torch.nonzero(dropped_slices, as_tuple=True)[0]
                if dropped_slice_indices.numel() > 0:
                    total_dropped_slices = dropped_slice_indices.numel()
                    num_slices_to_drop = self.rng.randint(0, min(7, total_dropped_slices))
                    if drop_type == 1:
                        m_end = int(dropped_slice_indices[-1].item() + 1)
                        m_start = int(max(dropped_slice_indices[0].item(), m_end - num_slices_to_drop))
                        if m_start < m_end:
                            img[0, m_start:m_end, :, :] = 0
                    elif drop_type == 2:
                        mid_point = int((dropped_slice_indices[0].item() + dropped_slice_indices[-1].item()) // 2)
                        half_slices = num_slices_to_drop // 2
                        m_start = int(max(0, mid_point - half_slices))
                        m_end = int(min(d, mid_point + (num_slices_to_drop - half_slices)))
                        if m_start < m_end:
                            img[0, m_start:m_end, :, :] = 0
                    else:
                        m_start = int(dropped_slice_indices[0].item())
                        m_end = int(min(m_start + num_slices_to_drop, dropped_slice_indices[-1].item() + 1))
                        if m_start < m_end:
                            img[0, m_start:m_end, :, :] = 0
        return img

class RandomShift3D:
    def __init__(self, max_shift=100, seed=42):
        self.max_shift = max_shift
        self.shifts = None
        self.rng = np.random.RandomState(seed)

    def __call__(self, image, *masks):
        if self.shifts is None:
            seg = masks[0].squeeze(0).numpy() if masks else None
            self._generate_shifts(seg)
        
        results = []
        for arr in [image] + list(masks):
            arr_np = arr.numpy().squeeze(0)
            shifted = shift(arr_np, self.shifts, mode='constant', cval=0)
            results.append(torch.from_numpy(shifted).unsqueeze(0))
        
        self.shifts = None
        return tuple(results)

    def _generate_shifts(self, segmentation=None):
        if segmentation is not None and np.any(segmentation):
            non_zero_coords = np.nonzero(segmentation)
            min_coords = [np.min(c) for c in non_zero_coords]
            max_coords = [np.max(c) for c in non_zero_coords]
            
            max_shift_up = min_coords
            max_shift_down = [segmentation.shape[i] - max_coords[i] - 1 
                             for i in range(3)]
            
            self.shifts = [
                self.rng.randint(-min(up, self.max_shift), min(down, self.max_shift) + 1)
                for up, down in zip(max_shift_up, max_shift_down)
            ]
        else:
            self.shifts = self.rng.randint(
                -self.max_shift, self.max_shift + 1, size=3
            )

class RandomShift3DWrapper:
    def __init__(self, max_shift=10, seed=42):
        self.shift_transform = RandomShift3D(max_shift, seed)
        self.current_seg = None
        self.current_complete_mask = None
        self.shifted_seg = None
        self.shifted_complete_mask = None

    def set_segmentation(self, seg):
        self.current_seg = seg

    def set_complete_mask(self, mask):
        self.current_complete_mask = mask

    def __call__(self, image):
        if self.current_seg is None or self.current_complete_mask is None:
            return image
        
        image, self.shifted_seg, self.shifted_complete_mask = self.shift_transform(
            image, 
            self.current_seg,
            self.current_complete_mask
        )
        
        return image

    def get_shifted_segmentation(self):
        return self.shifted_seg

    def get_shifted_complete_mask(self):
        return self.shifted_complete_mask

# Example usage:
# transform = transforms.Compose([
#     RandomShift3DWrapper(max_shift=10)
# ])
# train_dataset = NucCorrDataset(root_dir='path/to/data', split_file='train.txt', transform=transform)
# val_dataset = NucCorrDataset(root_dir='path/to/data', split_file='val.txt', transform=None) 