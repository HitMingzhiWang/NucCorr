import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpatialTransformer3D(nn.Module):
    """3D空间变换网络核心组件"""
    def __init__(self, mode='bilinear', padding_mode='border'):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, template, displacement_field):
        """
        应用变形场到模板
        template: [B, 1, D, H, W] 形状模板(SDF)
        displacement_field: [B, 3, D, H, W] 变形场(Δz, Δy, Δx)
        """
        B, _, D, H, W = template.shape
        
        # 创建归一化网格 [-1, 1]
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, D, device=template.device),
            torch.linspace(-1, 1, H, device=template.device),
            torch.linspace(-1, 1, W, device=template.device),
            indexing='ij'
        )
        
        # 添加批次维度 [B, D, H, W, 3]
        grid = torch.stack((x, y, z), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1, 1)
        
        # 将位移场转换为归一化偏移量
        # 位移场是绝对位移，需要转换为归一化空间的偏移
        disp_norm = displacement_field.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]
        disp_norm[..., 0] = disp_norm[..., 0] * (2 / (W - 1))  # x方向
        disp_norm[..., 1] = disp_norm[..., 1] * (2 / (H - 1))  # y方向
        disp_norm[..., 2] = disp_norm[..., 2] * (2 / (D - 1))  # z方向
        
        # 应用位移到网格
        warped_grid = grid + disp_norm
        
        # 使用grid_sample进行三线性插值
        warped_template = F.grid_sample(
            template, 
            warped_grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=True
        )
        
        return warped_template


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # 下采样连接
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class DeformationPredictionNetwork(nn.Module):
    def __init__(self, in_channels=3, feature_channels=32):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            ResidualBlock3D(feature_channels, feature_channels * 2),
            nn.MaxPool3d(2),
            
            ResidualBlock3D(feature_channels * 2, feature_channels * 4),
            nn.MaxPool3d(2),
            
            ResidualBlock3D(feature_channels * 4, feature_channels * 8),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(feature_channels * 8, feature_channels * 4, 
                              kernel_size=2, stride=2),
            ResidualBlock3D(feature_channels * 4, feature_channels * 4),
            
            nn.ConvTranspose3d(feature_channels * 4, feature_channels * 2, 
                              kernel_size=2, stride=2),
            ResidualBlock3D(feature_channels * 2, feature_channels * 2),
            
            nn.ConvTranspose3d(feature_channels * 2, feature_channels, 
                              kernel_size=2, stride=2),
            ResidualBlock3D(feature_channels, feature_channels),
        )
        
        # 输出变形场
        self.deformation_head = nn.Conv3d(feature_channels, 3, kernel_size=3, padding=1)
        
        nn.init.constant_(self.deformation_head.weight, 0)
        nn.init.constant_(self.deformation_head.bias, 0)

    def forward(self, x):
        # x: [B, 2, D, H, W] 输入图像+模板
        features = self.encoder(x)
        features = self.decoder(features)
        deformation = self.deformation_head(features)
        return deformation


class TETRIS_NucleiSegmentation(nn.Module):
    """基于TETRIS的3D细胞核分割模型"""
    def __init__(self, sdf_template):
        """
        sdf_template: [D, H, W] numpy数组，平均SDF模板
        """
        super().__init__()
        
        # 注册模板为缓冲区（不参与训练但随模型保存）
        self.register_buffer('template', torch.from_numpy(sdf_template).float().unsqueeze(0))
        
        # 变形场预测网络
        self.deformation_net = DeformationPredictionNetwork()
        
        # 空间变换器
        self.spatial_transformer = SpatialTransformer3D()
        
        # 可选的模板分类器（用于多模板场景）
        self.template_classifier = None
    
    def forward(self, image):
        """
        image: [B, 1, D, H, W] 输入3D图像
        返回: [B, 1, D, H, W] 变形后的SDF模板（分割结果）
        """
        # 验证模板与图像尺寸匹配
        assert image.shape[2:] == self.template.shape[1:], \
            f"模板尺寸{self.template.shape[1:]}≠图像尺寸{image.shape[2:]}"
        
        # 准备模板批次 [B, 1, D, H, W]
        template_batch = self.template.expand(image.size(0), 1, -1, -1, -1)
        
        # 双通道输入：图像 + 模板 [B, 3, D, H, W]
        inputs = torch.cat((image, template_batch), dim=1)
        
        # 预测变形场 [B, 3, D, H, W]
        deformation_field = self.deformation_net(inputs)
        
        # 应用变形到模板
        warped_template = self.spatial_transformer(template_batch, deformation_field)
        
        return warped_template
    
    def predict_binary_mask(self, image, threshold=0.0):
        """预测二值分割掩码"""
        with torch.no_grad():
            sdf = self(image)
            return (sdf > threshold).float()
    
    def get_deformation_field(self, image):
        """获取预测的变形场"""
        with torch.no_grad():
            # 准备模板批次
            template_batch = self.template.expand(image.size(0), 1, -1, -1, -1)
            inputs = torch.cat((image, template_batch), dim=1)
            return self.deformation_net(inputs)


""" class MultiTemplateTETRIS(nn.Module):
    def __init__(self, sdf_templates):
        super().__init__()
        
        # 注册模板为缓冲区
        self.templates = nn.ParameterList([
            nn.Parameter(torch.from_numpy(t).float().unsqueeze(0), requires_grad=False) 
            for t in sdf_templates
        ])
        self.num_templates = len(sdf_templates)
        
        # 变形场预测网络
        self.deformation_net = DeformationPredictionNetwork()
        
        # 空间变换器
        self.spatial_transformer = SpatialTransformer3D()
        
        # 模板分类器
        self.template_classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(32 * 8, 128),  # 输入通道需要匹配deformation_net的输出特征
            nn.ReLU(),
            nn.Linear(128, self.num_templates)
        )

    def forward(self, image):
        # 双通道输入：图像 + 空白模板（稍后替换）
        # 这里使用空白模板作为占位符
        blank = torch.zeros_like(image)
        inputs = torch.cat((image, blank), dim=1)
        
        # 提取特征用于模板分类
        features = self.deformation_net.encoder(inputs)
        features = torch.mean(features, dim=[2, 3, 4])  # 全局平均池化
        
        # 预测模板ID
        template_logits = self.template_classifier(features)
        template_probs = F.softmax(template_logits, dim=1)
        template_id = torch.argmax(template_probs, dim=1)
        
        # 选择模板
        selected_templates = []
        for i in range(image.size(0)):
            selected_templates.append(self.templates[template_id[i]])
        template_batch = torch.cat(selected_templates, dim=0)
        
        # 预测变形场
        deformation_field = self.deformation_net(inputs)
        
        # 应用变形到模板
        warped_template = self.spatial_transformer(template_batch, deformation_field)
        
        return warped_template, template_probs """