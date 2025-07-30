import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import rotate

class NeuronTransform:
    """3D数据增强变换"""
    def __init__(self, apply_rotation=True, apply_flip=True, noise_std=0.05):
        self.apply_rotation = apply_rotation
        self.apply_flip = apply_flip
        self.noise_std = noise_std
    
    def __call__(self, x):
        # x: [C, D, H, W]
        
        # 随机旋转 (0°, 90°, 180°, 270°)
        if self.apply_rotation:
            angle = np.random.choice([0, 90, 180, 270])
            if angle != 0:
                # 在高度和宽度维度旋转
                rotated = []
                for c in range(x.shape[0]):
                    # 对每个通道单独旋转
                    rotated_c = rotate(x[c].numpy(), angle, axes=(1, 2), reshape=False)
                    rotated.append(rotated_c)
                x = torch.tensor(np.stack(rotated))
        
        # 随机翻转 (深度、高度或宽度维度)
        if self.apply_flip:
            flip_dims = []
            if np.random.rand() > 0.5:
                flip_dims.append(1)  # 高度维度
            if np.random.rand() > 0.5:
                flip_dims.append(2)  # 宽度维度
            if np.random.rand() > 0.5 and x.shape[0] > 3:  # 确保深度维度存在
                flip_dims.append(0)
            if flip_dims:
                x = torch.flip(x, dims=flip_dims)
        
        # 添加高斯噪声 (仅对图像通道)
        if self.noise_std > 0:
            noise = torch.randn_like(x[0]) * self.noise_std
            x[0] = x[0] + noise
        
        return x