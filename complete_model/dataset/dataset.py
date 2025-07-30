import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms
import tifffile
from scipy.ndimage import shift
import torch.nn.functional as F

class NucCorrDataset(Dataset):
    def __init__(self, root_dir, split_file, transform=None, img_size=(128, 128, 128), train=True, seed=42, shift_augment=True):
        """
        Args:
            root_dir (string): Directory with all the images and segmentation masks.
            split_file (string): Path to the txt file containing the list of files for this split.
            transform (callable, optional): Optional transform to be applied on a sample.
            img_size (tuple): Size of the 3D images (depth, height, width)
            train (bool): Whether this is training set
            seed (int): Random seed for reproducibility
        """

        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.train = train
        self.seed = seed
        self.shift_augment = shift_augment and train
        if self.shift_augment:
            self.shift_transform = PyTorchShift3D(max_shift=128, p=0.8, seed=seed)
        # 设置随机种子
        self.rng = np.random.RandomState(seed)
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(seed)
        
        # Read the split file
        with open(split_file, 'r') as f:
            self.file_list = [line.strip() + '.tiff' for line in f.readlines()]
        
        # Get all image and segmentation files
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
        """获取当前随机状态"""
        return {
            'numpy': self.rng.get_state(),
            'torch': self.torch_rng.get_state()
        }
    
    def _set_random_state(self, state):
        """设置随机状态"""
        self.rng.set_state(state['numpy'])
        self.torch_rng.set_state(state['torch'])
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Load image and segmentation
        img_path = os.path.join(self.root_dir, 'img', self.img_files[idx])
        seg_path = os.path.join(self.root_dir, 'seg', self.seg_files[idx])
        
        image = tifffile.imread(img_path)
        segmentation = tifffile.imread(seg_path)
        
        # Ensure correct shape
        assert image.shape == self.img_size, f"Image shape {image.shape} doesn't match expected shape {self.img_size}"
        assert segmentation.shape == self.img_size, f"Segmentation shape {segmentation.shape} doesn't match expected shape {self.img_size}"
        
        # Convert segmentation to binary (0 and 1)
        segmentation = (segmentation > 0).astype(np.uint8)
        
        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        segmentation = torch.from_numpy(segmentation).long()
        
        # Add channel dimension
        image = image.unsqueeze(0)  # Add channel dimension for image
        segmentation = segmentation.unsqueeze(0)
        
        # Normalize image
        image = image / 255.0
        
        # Save the complete mask BEFORE any augmentations
        complete_mask = segmentation.clone()
        
        if True:
            # First apply segmentation augmentations and get dropped regions
            segmentation, dropped_regions, drop_type, dropped_blocks = self._apply_seg_augmentations(segmentation)
            
            # Then apply image augmentations using the dropped regions and drop type
            image = self._apply_img_augmentations(image, segmentation, dropped_regions, drop_type, dropped_blocks)
            
            # Finally apply transformations if any - use complete_mask for shift calculation
            if self.shift_augment:
                image, segmentation, complete_mask = self.shift_transform(image, segmentation, complete_mask)
        
        return segmentation, image, complete_mask
    
    def _apply_seg_augmentations(self, seg):
        """Apply segmentation augmentations with vectorized operations"""
        # Create a mask to track dropped regions
        dropped_regions = torch.zeros_like(seg.squeeze(0))
        drop_type = random.choice([-1, 0, 1, 2, 3])   # -1: no drop, 0: patch drop, 1: slice drop from start, 2: slice drop from middle, 3: slice drop from end
        dropped_blocks = []  # Store the blocks that were dropped
        
        if self.rng.random() < 0.9:  # 90% chance to apply augmentation
            # Get foreground coordinates
            foreground = seg.squeeze(0).bool()
            if not foreground.any():
                return seg, dropped_regions, drop_type, dropped_blocks
            
            # Choose augmentation type
            drop_type = self.rng.randint(0, 4)
            if drop_type == 0:  # Random patch drop
                block_size = (8, 8, 8)
                max_dropout_ratio = 0.9
                
                d, h, w = foreground.shape
                if d >= block_size[0] and h >= block_size[1] and w >= block_size[2]:
                    # 使用向量化操作创建块掩码
                    # 计算块的数量
                    d_blocks = d // block_size[0]
                    h_blocks = h // block_size[1]
                    w_blocks = w // block_size[2]
                    
                    # 调整前景以匹配块网格
                    foreground_grid = foreground[:d_blocks*block_size[0], :h_blocks*block_size[1], :w_blocks*block_size[2]]
                    
                    # 重塑为块
                    reshaped = foreground_grid.reshape(
                        d_blocks, block_size[0],
                        h_blocks, block_size[1],
                        w_blocks, block_size[2]
                    )
                    
                    # 检查哪些块包含前景
                    block_has_foreground = reshaped.any(dim=(1, 3, 5))
                    
                    # 获取候选块索引
                    candidate_blocks = torch.nonzero(block_has_foreground, as_tuple=False)
                    num_candidates = candidate_blocks.size(0)
                    
                    if num_candidates > 0:
                        dropout_ratio = self.rng.uniform(0, max_dropout_ratio)
                        num_drop = int(num_candidates * dropout_ratio)
                        num_drop = max(0, min(num_drop, num_candidates))
                        
                        if num_drop > 0:
                            # 创建丢弃掩码
                            drop_mask = torch.zeros_like(block_has_foreground)
                            
                            # 随机选择要丢弃的块
                            drop_indices = torch.randperm(num_candidates, generator=self.torch_rng)[:num_drop]
                            for idx in drop_indices:
                                dz, dy, dx = candidate_blocks[idx].tolist()
                                drop_mask[dz, dy, dx] = 1
                                
                            
                                z_start = dz * block_size[0]
                                y_start = dy * block_size[1]
                                x_start = dx * block_size[2]
                                z_end = min(z_start + block_size[0], d)
                                y_end = min(y_start + block_size[1], h)
                                x_end = min(x_start + block_size[2], w)
                                dropped_blocks.append((z_start, z_end, y_start, y_end, x_start, x_end))
                            
                            # 扩展丢弃掩码到完整尺寸
                            expanded_drop_mask = drop_mask.repeat_interleave(block_size[0], dim=0)
                            expanded_drop_mask = expanded_drop_mask.repeat_interleave(block_size[1], dim=1)
                            expanded_drop_mask = expanded_drop_mask.repeat_interleave(block_size[2], dim=2)
                            
                            # 确保掩码大小匹配
                            expanded_drop_mask = expanded_drop_mask[:d, :h, :w]
                            
                            # 应用丢弃
                            seg[0, expanded_drop_mask] = 0
                            dropped_regions[expanded_drop_mask] = 1
            
            else:  # Z-axis slice drop
                z_coords = torch.any(foreground, dim=(1, 2))
                nonzero_z = torch.nonzero(z_coords, as_tuple=True)[0]
                
                if nonzero_z.numel() > 0:
                    z_start = nonzero_z.min().item()
                    z_end = nonzero_z.max().item()
                    mask_z_length = z_end - z_start + 1
                    max_mask_length = int(mask_z_length * 0.9)
                    
                    if max_mask_length > 0:
                        if drop_type == 1:  # Drop from start
                            mask_length = self.rng.randint(0, max_mask_length)
                            m_start = z_start
                            m_end = min(z_start + mask_length, z_end + 1)
                            if m_start < m_end:
                                seg[0, m_start:m_end, :, :] = 0
                                dropped_regions[m_start:m_end, :, :] = 1
                                
                        elif drop_type == 2:  # Drop from middle
                            mid_point = (z_start + z_end) // 2
                            mask_length = self.rng.randint(0, max_mask_length // 2)
                            m_start = max(z_start, mid_point - mask_length)
                            m_end = min(z_end + 1, mid_point + mask_length)
                            if m_start < m_end:
                                seg[0, m_start:m_end, :, :] = 0
                                dropped_regions[m_start:m_end, :, :] = 1
                                
                        else:  # Drop from end
                            mask_length = self.rng.randint(0, max_mask_length)
                            m_start = max(z_start, z_end - mask_length + 1)
                            m_end = z_end + 1
                            if m_start < m_end:
                                seg[0, m_start:m_end, :, :] = 0
                                dropped_regions[m_start:m_end, :, :] = 1
        
        return seg, dropped_regions, drop_type, dropped_blocks
    
    def _apply_img_augmentations(self, img, seg, dropped_regions, drop_type, dropped_blocks):
        """Apply image augmentations with optimized operations"""
        # 90% chance to apply image drop
        if self.rng.random() < 0.9:  
            # 随机决定使用0还是平均像素值
            use_mean_value = self.rng.random() < 0.5
            mean_val = img.mean().item() if use_mean_value else 0
            
            d, h, w = dropped_regions.shape
            
            # Apply the same type of drop as segmentation
            if drop_type == 0 and dropped_blocks:  # Patch drop
                # 随机选择90%的丢弃块用于图像drop
                num_blocks = len(dropped_blocks)
                num_drop = max(1, int(num_blocks * 0.9))
                selected_indices = torch.randperm(num_blocks, generator=self.torch_rng)[:num_drop]
                
                # 创建图像丢弃掩码
                img_drop_mask = torch.zeros_like(dropped_regions, dtype=torch.bool)
                for idx in selected_indices:
                    z_start, z_end, y_start, y_end, x_start, x_end = dropped_blocks[idx]
                    img_drop_mask[z_start:z_end, y_start:y_end, x_start:x_end] = True
                
                # 一次性应用丢弃
                img[0, img_drop_mask] = mean_val
            
            elif drop_type > 0:  # Z-axis slice drop
                # Find slices that are in dropped regions
                dropped_slices = torch.any(dropped_regions, dim=(1, 2))
                dropped_slice_indices = torch.nonzero(dropped_slices, as_tuple=True)[0]
                
                if dropped_slice_indices.numel() > 0:
                    # Randomly select number of slices to drop between 1 and total slices
                    total_dropped_slices = dropped_slice_indices.numel()
                    num_slices_to_drop = self.rng.randint(0, min(7, total_dropped_slices))
                    
                    if num_slices_to_drop > 0:
                        # Apply opposite direction drop for image
                        if drop_type == 1:  # If seg drops from start, img drops from end
                            m_end = dropped_slice_indices[-1] + 1
                            m_start = max(dropped_slice_indices[0], m_end - num_slices_to_drop)
                            if m_start < m_end:
                                img[0, m_start:m_end, :, :] = mean_val
                                
                        elif drop_type == 2:  # Drop from middle (keep the same)
                            mid_point = (dropped_slice_indices[0] + dropped_slice_indices[-1]) // 2
                            half_slices = num_slices_to_drop // 2
                            m_start = max(0, mid_point - half_slices)
                            m_end = min(d, mid_point + (num_slices_to_drop - half_slices))
                            if m_start < m_end:
                                img[0, m_start:m_end, :, :] = mean_val
                                
                        else:  # If seg drops from end, img drops from start
                            m_start = dropped_slice_indices[0]
                            m_end = min(m_start + num_slices_to_drop, dropped_slice_indices[-1] + 1)
                            if m_start < m_end:
                                img[0, m_start:m_end, :, :] = mean_val
        
        return img

class PyTorchShift3D:
    def __init__(self, max_shift=128, p=1, seed=42):
        self.max_shift = max_shift
        self.p = p  # 应用变换的概率
        self.rng = np.random.RandomState(seed)

    def __call__(self, *tensors):
        if random.random() > self.p:
            return tensors if len(tensors) > 1 else tensors[0]

        # 获取参考 segmentation，用于限制偏移量
        ref_seg = None
        for tensor in tensors:
            if tensor.dtype == torch.long and tensor.dim() == 4:
                ref_seg = tensor.squeeze(0).cpu().numpy()
                break
        if ref_seg is None and len(tensors) > 0:
            last_tensor = tensors[-1]
            if last_tensor.dtype == torch.long and last_tensor.dim() == 4:
                ref_seg = last_tensor.squeeze(0).cpu().numpy()
    
        shifts = self._generate_shifts(ref_seg)
        # print("Voxel shifts (z,y,x):", shifts)

        results = []
        for tensor in tensors:
            is_batched = tensor.dim() == 4  # (C, D, H, W)
            if not is_batched:
                raise ValueError("Input tensor must be batched (C, D, H, W)")

            results.append(self._apply_shift(tensor, shifts))

        return tuple(results) if len(results) > 1 else results[0]

    def _apply_shift(self, tensor, shifts):
        """对单个 tensor 应用整数平移"""
        c, d, h, w = tensor.shape
        shift_z, shift_y, shift_x = shifts

        # 计算 pad 和 crop 的索引
        pad = [
            max(shift_x, 0), max(-shift_x, 0),
            max(shift_y, 0), max(-shift_y, 0),
            max(shift_z, 0), max(-shift_z, 0),
        ]
        tensor = F.pad(tensor, pad, mode='constant', value=0)

        # crop to original size
        start_z = max(-shift_z, 0)
        start_y = max(-shift_y, 0)
        start_x = max(-shift_x, 0)

        shifted = tensor[:, start_z:start_z + d, start_y:start_y + h, start_x:start_x + w]
        return shifted

    def _generate_shifts(self, segmentation=None):
        if segmentation is not None and np.any(segmentation):
            non_zero_coords = np.nonzero(segmentation)
            min_coords = [np.min(c) for c in non_zero_coords]
            max_coords = [np.max(c) for c in non_zero_coords]
            shape = segmentation.shape

            shift_range = []
            for dim in range(3):
                shift_min = -min_coords[dim]
                shift_max = shape[dim] - 1 - max_coords[dim]
                shift_min = max(shift_min, -self.max_shift)
                shift_max = min(shift_max, self.max_shift)
                shift_range.append(self.rng.randint(shift_min, shift_max + 1))

            return shift_range
        else:
            return self.rng.randint(-self.max_shift, self.max_shift + 1, size=3).tolist()