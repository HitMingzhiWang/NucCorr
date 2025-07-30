import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import generate_model  # 注意替换成你自己的ResNet模块路径


def euler_angles_to_rotation_matrix(euler, order='xyz'):
    """
    将 batch 的欧拉角转换为旋转矩阵
    Args:
        euler: (B, 3) in radians
        order: rotation order, like 'xyz' or 'zyx'
    Returns:
        rot_matrix: (B, 3, 3)
    """
    B = euler.size(0)
    c = torch.cos(euler)
    s = torch.sin(euler)

    ones = torch.ones(B, device=euler.device)
    zeros = torch.zeros(B, device=euler.device)

    def build_matrix(axis):
        if axis == 'x':
            return torch.stack([
                torch.stack([ones, zeros, zeros], dim=1),
                torch.stack([zeros, c[:, 0], -s[:, 0]], dim=1),
                torch.stack([zeros, s[:, 0],  c[:, 0]], dim=1)
            ], dim=1)
        elif axis == 'y':
            return torch.stack([
                torch.stack([c[:, 1], zeros, s[:, 1]], dim=1),
                torch.stack([zeros, ones, zeros], dim=1),
                torch.stack([-s[:, 1], zeros, c[:, 1]], dim=1)
            ], dim=1)
        elif axis == 'z':
            return torch.stack([
                torch.stack([c[:, 2], -s[:, 2], zeros], dim=1),
                torch.stack([s[:, 2],  c[:, 2], zeros], dim=1),
                torch.stack([zeros, zeros, ones], dim=1)
            ], dim=1)
        else:
            raise ValueError(f"Invalid axis {axis}")

    # Apply rotations in specified order
    rot = torch.eye(3, device=euler.device).unsqueeze(0).repeat(B, 1, 1)  # identity
    for axis in order:
        rot_axis = build_matrix(axis)
        rot = torch.bmm(rot, rot_axis)
    return rot


class ShapeAndAffinePredictor(nn.Module):
    def __init__(self, feature_dim=512, latent_dim=30):
        super().__init__()
        self.shape_head = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.affine_head = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 9)  # 3 translation + 3 euler + 3 scale
        )

    def forward(self, feat):
        beta = self.shape_head(feat)           # (B, latent_dim)
        params = self.affine_head(feat)        # (B, 9)

        t = params[:, :3]                      # Translation (B, 3)
        r = params[:, 3:6]                     # Euler angles (B, 3)
        s = params[:, 6:9]                     # Scaling factors (B, 3)

        R = euler_angles_to_rotation_matrix(r, order='xyz')  # (B, 3, 3)
        S = torch.diag_embed(s)                               # (B, 3, 3)

        linear_matrix = torch.bmm(R, S)  # 先旋转后缩放

        affine_matrix = torch.zeros(feat.size(0), 4, 4, device=feat.device)
        affine_matrix[:, :3, :3] = linear_matrix
        affine_matrix[:, :3, 3] = t
        affine_matrix[:, 3, 3] = 1.0

        return beta, affine_matrix


class ASMPointPredictor(nn.Module):
    def __init__(self, p_mean: torch.Tensor, p_components: torch.Tensor):
        super().__init__()
        self.p_mean = nn.Parameter(p_mean.clone(), requires_grad=False)         # (N, 3)
        self.components = nn.Parameter(p_components.clone(), requires_grad=False)  # (latent_dim, 3N)

        self.encoder = generate_model(model_depth=34)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.head = ShapeAndAffinePredictor(feature_dim=512, latent_dim=p_components.shape[0])

    def forward(self, x):
        """
        Args:
            x: (B, 1, D, H, W) - 输入3D图像
        Returns:
            pred_points: (B, N, 3)
            beta: (B, latent_dim)
            theta: (B, 4, 4)
        """
        B = x.size(0)
        feat = self.encoder(x)                               # (B, 512, D', H', W')
        feat = self.global_pool(feat).view(B, -1)            # (B, 512)

        beta, theta = self.head(feat)                        # (B, latent), (B, 4, 4)

        # Shape 重建
        p_shape = self.p_mean.view(1, -1).repeat(B, 1) + torch.matmul(beta, self.components)  # (B, 3N)
        p_shape = p_shape.view(B, -1, 3)                     # (B, N, 3)

        # 仿射变换
        ones = torch.ones(B, p_shape.size(1), 1, device=x.device)
        p_homo = torch.cat([p_shape, ones], dim=2)           # (B, N, 4)
        pred_points = torch.bmm(p_homo, theta.transpose(1, 2))  # (B, N, 4)
        pred_points = pred_points[:, :, :3]                  # 去掉齐次坐标

        return pred_points
