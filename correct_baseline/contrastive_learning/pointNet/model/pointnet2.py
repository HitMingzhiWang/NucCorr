import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG


class PointNet2Contrastive(nn.Module):
    def __init__(self, input_dim=3, embed_dim=16, num_classes=2, use_embeddings=True):
        """
        PointNet2对比学习模型
        
        Args:
            input_dim: 点云坐标维度 (3)
            embed_dim: 嵌入特征维度 (从3DCNN提取的特征维度)
            num_classes: 分类数量 (2 for 正负样本)
            use_embeddings: 是否使用嵌入特征
        """
        super().__init__()
        
        self.use_embeddings = use_embeddings
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # 计算输入特征维度
        if self.use_embeddings:
            self.feature_dim = input_dim + embed_dim  # 3 + embed_dim
        else:
            self.feature_dim = input_dim  # 只有3维坐标
        
        print(f"PointNet2 Input: {self.feature_dim} dimensions (xyz: {input_dim} + embeddings: {embed_dim if use_embeddings else 0})")

        # SA1 with MSG - 处理concat后的特征
        self.SA1 = PointnetSAModuleMSG(
            npoint=512,
            radii=[0.1, 0.2, 0.4],
            nsamples=[16, 32, 128],
            mlps=[
                [self.feature_dim, 32, 32, 64],    # MLP for radius 0.1
                [self.feature_dim, 64, 64, 128],   # MLP for radius 0.2
                [self.feature_dim, 64, 64, 128]    # MLP for radius 0.4
            ],
            use_xyz=True  # 使用xyz坐标信息
        )
        # 输出特征: 64 + 128 + 128 = 320 channels

        # SA2 with MSG
        self.SA2 = PointnetSAModuleMSG(
            npoint=128,
            radii=[0.2, 0.4, 0.8],
            nsamples=[32, 64, 128],
            mlps=[
                [320, 64, 64, 128],   # MLP for radius 0.2
                [320, 128, 128, 256], # MLP for radius 0.4
                [320, 128, 128, 256]  # MLP for radius 0.8
            ],
            use_xyz=True
        )
        # 输出特征: 128 + 256 + 256 = 640 channels

        # SA3 - 全局特征聚合
        self.SA3 = PointnetSAModule(
            npoint=None,
            radius=None,
            nsample=None,
            mlp=[640, 256, 512, 1024],
            use_xyz=True
        )

        # 分类器 - 用于对比学习
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),  # 特征维度
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )
        
        # 对比学习投影头
        self.projection_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128)  # 投影到128维用于对比学习
        )

    def forward(self, pointcloud, return_features=False):
        """
        前向传播
        
        Args:
            pointcloud: B x N x (3+embed_dim) 或 B x N x 3
            return_features: 是否返回中间特征用于对比学习
            
        Returns:
            classification_output: B x num_classes
            projection_features: B x 128 (如果return_features=True)
        """
        xyz = pointcloud[:, :, :3]  # 提取xyz坐标 B x N x 3
        features = pointcloud  # 完整特征 B x N x (3+embed_dim)
        
        B, N, _ = xyz.shape

        # 准备输入
        l_xyz = xyz.contiguous()
        l_features = features.transpose(1, 2).contiguous()  # (B, feature_dim, N)

        # SA1 (MSG)
        l_xyz_1, l_features_1 = self.SA1(l_xyz, l_features)  # l_features_1: (B, 320, 512)

        # SA2 (MSG)
        l_xyz_2, l_features_2 = self.SA2(l_xyz_1, l_features_1)  # l_features_2: (B, 640, 128)

        # SA3 (global SA)
        l_xyz_3, l_features_3 = self.SA3(l_xyz_2, l_features_2)  # l_features_3: (B, 1024, 1)

        # 全局特征
        global_features = l_features_3.view(B, -1)  # (B, 1024)
        
        # 分类输出
        classification_output = self.classifier(global_features)
        
        if return_features:
            # 对比学习投影特征
            projection_features = self.projection_head(global_features)
            return classification_output, projection_features
        else:
            return classification_output


class PointNet2Classification(nn.Module):
    """原始分类模型 - 保持兼容性"""
    def __init__(self, num_classes=1, input_dim=3):
        super().__init__()
        
        self.input_dim = input_dim
        print(f"PointNet2 Classification Input: {input_dim} dimensions")

        # SA1 with MSG
        self.SA1 = PointnetSAModuleMSG(
            npoint=512,
            radii=[0.1, 0.2, 0.4],
            nsamples=[16, 32, 128],
            mlps=[
                [input_dim, 32, 32, 64],
                [input_dim, 64, 64, 128],
                [input_dim, 64, 64, 128]
            ],
            use_xyz=False
        )

        # SA2 with MSG
        self.SA2 = PointnetSAModuleMSG(
            npoint=128,
            radii=[0.2, 0.4, 0.8],
            nsamples=[32, 64, 128],
            mlps=[
                [320, 64, 64, 128],
                [320, 128, 128, 256],
                [320, 128, 128, 256]
            ],
            use_xyz=True
        )

        # SA3
        self.SA3 = PointnetSAModule(
            npoint=None,
            radius=None,
            nsample=None,
            mlp=[640, 256, 512, 1024],
            use_xyz=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, pointcloud):
        """
        Input: B x N x input_dim
        Output: B x num_classes
        """
        xyz = pointcloud
        B, N, _ = xyz.shape

        l_xyz = xyz.contiguous()
        l_features = xyz.transpose(1, 2).contiguous()

        # SA1 (MSG)
        l_xyz_1, l_features_1 = self.SA1(l_xyz, l_features)

        # SA2 (MSG)
        l_xyz_2, l_features_2 = self.SA2(l_xyz_1, l_features_1)

        # SA3 (global SA)
        l_xyz_3, l_features_3 = self.SA3(l_xyz_2, l_features_2)

        # Flatten the global features for classification
        x = l_features_3.view(B, -1)
        x = self.classifier(x)
        return x


