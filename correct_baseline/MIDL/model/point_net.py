import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG


class PointNet2Classification(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # SA1 with MSG
        self.SA1 = PointnetSAModuleMSG(
            npoint=512,
            radii=[0.1, 0.2, 0.4],  # Multiple radii for MSG
            nsamples=[16, 32, 128], # Samples per radius
            # mlp for each scale: [input_features, out_1, out_2, ...]
            # For the first SA layer, input features are 3 (XYZ), as we're passing xyz as features
            mlps=[
                [3, 32, 32, 64],    # MLP for radius 0.1
                [3, 64, 64, 128],   # MLP for radius 0.2
                [3, 64, 64, 128]    # MLP for radius 0.4
            ],
            use_xyz=False # Still False, as we're explicitly feeding XYZ as initial features
        )
        # The output features from SA1 will be the sum of the last MLP outputs: 64 + 128 + 128 = 320 channels

        # SA2 with MSG
        self.SA2 = PointnetSAModuleMSG(
            npoint=128,
            radii=[0.2, 0.4, 0.8],
            nsamples=[32, 64, 128],
            # The input features here are the concatenated features from the previous SA layer (320 from SA1)
            mlps=[
                [320, 64, 64, 128],   # MLP for radius 0.2
                [320, 128, 128, 256], # MLP for radius 0.4
                [320, 128, 128, 256]  # MLP for radius 0.8
            ],
            use_xyz=True # Use XYZ here, as l_features will be from previous SA layer, not XYZ
        )
        # The output features from SA2 will be the sum of the last MLP outputs: 128 + 256 + 256 = 640 channels

        # SA3 remains a regular SA module for global feature aggregation
        # because npoint=None implies a single centroid (global feature).
        # Its input features will be the concatenated features from SA2 (640 channels).
        self.SA3 = PointnetSAModule(
            npoint=None,
            radius=None,
            nsample=None,
            mlp=[640, 256, 512, 1024], # Input features are 640 from SA2
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
            nn.Linear(256, num_classes), # Output is 2 channels for num_classes=2
            nn.Sigmoid() # Add Sigmoid here
        )

    def forward(self, pointcloud):
        """
        Input: B x N x 3
        Output: B x 2
        """
        xyz = pointcloud
        B, N, _ = xyz.shape

        # Prepare initial features: XYZ as features for the first SA layer
        # Transpose to B x C x N format for PointNet++ operations
        l_xyz = xyz.contiguous()
        l_features = xyz.transpose(1, 2).contiguous() # (B, 3, N)

        # Pass through SA layers
        # SA1 (MSG)
        l_xyz_1, l_features_1 = self.SA1(l_xyz, l_features) # l_features_1 will be (B, 320, 512)

        # SA2 (MSG)
        # The output features from SA1 (l_features_1) become input features for SA2
        l_xyz_2, l_features_2 = self.SA2(l_xyz_1, l_features_1) # l_features_2 will be (B, 640, 128)

        # SA3 (global SA)
        # The output features from SA2 (l_features_2) become input features for SA3
        l_xyz_3, l_features_3 = self.SA3(l_xyz_2, l_features_2) # l_features_3 will be (B, 1024, 1)

        # Flatten the global features for classification
        x = l_features_3.view(B, -1) # Reshape to (B, 1024)
        x = self.classifier(x)
        return x