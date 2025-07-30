import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import tifffile
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

def mask_to_sdf(mask):
    """
    将二值 3D mask (支持1*D*H*W或D*H*W) 转为归一化 SDF (Signed Distance Function).
    输出范围为 [-1, 1]，0为mask边界。
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    # 自动去掉多余的维度
    mask = np.squeeze(mask)
    assert mask.ndim == 3, f"Only supports 3D mask, got shape {mask.shape}"
    from scipy.ndimage import distance_transform_edt

    # 前景距离（背景上到前景的距离）
    dist_out = distance_transform_edt(mask == 0)
    # 背景距离（前景上到背景的距离）
    dist_in = distance_transform_edt(mask == 1)
    # SDF = 正的背景距离 - 负的前景距离
    sdf = dist_out - dist_in

    # 归一化到 [-1, 1]
    max_abs = np.max(np.abs(sdf))
    if max_abs > 0:
        sdf = sdf / max_abs
    return sdf.astype(np.float32)

class NucCorrDataset(Dataset):
    def __init__(self, root_dir, split_file, transform=None, img_size=(128, 128, 128), train=True, seed=42):
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
        
        # Save the complete mask
        complete_mask = segmentation.clone()
        
        if True:
            # First apply segmentation augmentations and get dropped regions
            segmentation, dropped_regions, drop_type, dropped_blocks = self._apply_seg_augmentations(segmentation)
            
            # Then apply image augmentations using the dropped regions and drop type
            image = self._apply_img_augmentations(image, segmentation, dropped_regions, drop_type, dropped_blocks)
            
            # Finally apply transformations if any
            if self.transform:
                # 单次调用完成所有偏移
                image, segmentation, complete_mask = self.transform(image),self.transform(segmentation),self.transform(complete_mask)
        sdf = mask_to_sdf(complete_mask)
        return segmentation, image, complete_mask, sdf
    
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
                                
                                # 记录丢弃的块
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
                img[0, img_drop_mask] = 0
            
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
                                img[0, m_start:m_end, :, :] = 0
                                
                        elif drop_type == 2:  # Drop from middle (keep the same)
                            mid_point = (dropped_slice_indices[0] + dropped_slice_indices[-1]) // 2
                            half_slices = num_slices_to_drop // 2
                            m_start = max(0, mid_point - half_slices)
                            m_end = min(d, mid_point + (num_slices_to_drop - half_slices))
                            if m_start < m_end:
                                img[0, m_start:m_end, :, :] = 0
                                
                        else:  # If seg drops from end, img drops from start
                            m_start = dropped_slice_indices[0]
                            m_end = min(m_start + num_slices_to_drop, dropped_slice_indices[-1] + 1)
                            if m_start < m_end:
                                img[0, m_start:m_end, :, :] = 0
        
        return img

class PyTorchShift3D:
    """使用PyTorch的grid_sample进行3D偏移，比scipy.ndimage.shift更快"""
    def __init__(self, max_shift=100, seed=42):
        self.max_shift = max_shift
        self.shifts = None
        self.rng = np.random.RandomState(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self, image, *masks):
        # 确定设备
        image = image.to(self.device)
        masks = [m.to(self.device) for m in masks]
        
        # 生成偏移量
        if self.shifts is None:
            seg = masks[0].squeeze(0).cpu().numpy() if masks else None
            self._generate_shifts(seg)
        
        # 创建网格
        d, h, w = image.shape[-3:]
        grid_z, grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, d, device=self.device),
            torch.linspace(-1, 1, h, device=self.device),
            torch.linspace(-1, 1, w, device=self.device),
            indexing='ij'
        )
        
        # 应用偏移（归一化偏移量）
        shift_z, shift_y, shift_x = self.shifts
        grid_z = grid_z + 2 * shift_z / d
        grid_y = grid_y + 2 * shift_y / h
        grid_x = grid_x + 2 * shift_x / w
        
        # 组合网格
        grid = torch.stack((grid_x, grid_y, grid_z), dim=-1).unsqueeze(0)
        
        # 对图像和每个mask进行变换
        results = []
        for arr in [image] + list(masks):
            # 对于分割mask使用最近邻插值，对于图像使用双线性插值
            mode = 'nearest' if arr.shape[0] == 1 and arr.dtype == torch.long else 'bilinear'
            aligned = 'align_corners' if mode == 'bilinear' else None
            
            # 使用grid_sample进行变换
            shifted = F.grid_sample(
                arr.float(), 
                grid, 
                mode=mode,
                padding_mode='zeros',
                align_corners=aligned
            )
            
            # 恢复原始数据类型
            if arr.dtype == torch.long:
                shifted = shifted.long()
                
            results.append(shifted.squeeze(0).cpu())
        
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
        self.shift_transform = PyTorchShift3D(max_shift, seed)
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