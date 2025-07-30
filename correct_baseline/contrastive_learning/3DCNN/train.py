import os
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import random
import time
from tqdm import tqdm
import json
import gc

# ====================== 数据集类 (已修改) ======================
class NeuronConnectivityDataset(Dataset):
    """神经元连接数据集类，支持平移数据增强"""
    def __init__(self, data_root: str, img_dir: str, seg_dir: str, 
                 split_file: str, volume_size: tuple = (128, 128, 128),
                 augment: bool = False, max_translation: int = 10): # 新增 augment 和 max_translation 参数
        """
        初始化数据集
        data_root: 数据根目录
        img_dir: 图像体积目录名
        seg_dir: 分割掩码目录名
        split_file: 划分文件路径
        volume_size: 体积尺寸 (D, H, W)
        augment: 是否启用数据增强 (平移)
        max_translation: 最大平移像素值 (在每个维度上，正负 max_translation 之间)
        """
        self.data_root = data_root
        self.img_dir = os.path.join(data_root, img_dir)
        self.seg_dir = os.path.join(data_root, seg_dir)
        self.volume_size = volume_size
        self.augment = augment
        self.max_translation = max_translation
        
        # 读取划分文件
        with open(os.path.join(data_root, split_file), 'r') as f:
            self.volume_list = [line.strip() for line in f]
        
        print(f"数据集初始化完成，共加载 {len(self.volume_list)} 个体积")
        if self.augment:
            print(f"数据增强已启用: 最大平移 {self.max_translation} 像素")
    
    def __len__(self) -> int:
        """每个体积生成1个样本 (包含正负对信息)"""
        return len(self.volume_list)
    
    def __getitem__(self, idx: int) -> tuple:
        """获取样本：原始图像 + 各自的掩码 + 标签"""
        vol_name = self.volume_list[idx]
        
        # 加载图像和分割
        img = tifffile.imread(os.path.join(self.img_dir, vol_name)).astype(np.float32)
        seg = tifffile.imread(os.path.join(self.seg_dir, vol_name))
        
        # 确保体积尺寸正确
        if img.shape != self.volume_size:
            raise ValueError(f"体积尺寸不匹配: {img.shape} != {self.volume_size}")
        
        # ==================== 应用数据增强 (平移) ====================
        if self.augment:
            D, H, W = self.volume_size
            
            # 随机生成平移量
            # np.random.randint(low, high+1) 生成 [low, high] 范围内的整数
            shift_d = np.random.randint(-self.max_translation, self.max_translation + 1)
            shift_h = np.random.randint(-self.max_translation, self.max_translation + 1)
            shift_w = np.random.randint(-self.max_translation, self.max_translation + 1)

            # 为了进行平移裁剪，我们先对图像和分割掩码进行填充
            # 填充大小为 max_translation
            padded_shape = (D + 2 * self.max_translation,
                            H + 2 * self.max_translation,
                            W + 2 * self.max_translation)
            
            padded_img = np.zeros(padded_shape, dtype=img.dtype)
            padded_seg = np.zeros(padded_shape, dtype=seg.dtype)

            # 将原始数据放置在填充后的中心区域
            padded_img[self.max_translation : self.max_translation + D,
                       self.max_translation : self.max_translation + H,
                       self.max_translation : self.max_translation + W] = img
            
            padded_seg[self.max_translation : self.max_translation + D,
                       self.max_translation : self.max_translation + H,
                       self.max_translation : self.max_translation + W] = seg

            # 计算裁剪的起始坐标
            # 这个起始坐标加上 volume_size 后，将从填充后的体积中裁剪出原始大小的体积，
            # 且这个裁剪区域会根据 shift_d/h/w 进行偏移。
            start_d = self.max_translation + shift_d
            start_h = self.max_translation + shift_h
            start_w = self.max_translation + shift_w

            # 执行裁剪操作，确保图像和分割掩码应用相同的平移
            img = padded_img[start_d : start_d + D,
                             start_h : start_h + H,
                             start_w : start_w + W]
            
            seg = padded_seg[start_d : start_d + D,
                             start_h : start_h + H,
                             start_w : start_w + W]
        # ==========================================================
        
        # 根据论文的简化，这里硬编码了用于正负样本的片段ID
        # 实际应用中，这些ID会根据连接注释动态生成
        seg_id_query = 1
        seg_id_pos = 2
        seg_id_neg = 3

        # 获取各个片段的二值掩码 (这里获取的掩码是经过平移增强后的seg中提取的)
        mask_query = self._get_mask(seg, seg_id_query) # M_A
        mask_pos = self._get_mask(seg, seg_id_pos)     # M_B (positive for M_A)
        mask_neg = self._get_mask(seg, seg_id_neg)     # M_C (negative for M_A)
        
        # 转换为PyTorch张量
        img_tensor = torch.from_numpy(img).unsqueeze(0) # [1, D, H, W]
        mask_query_tensor = torch.from_numpy(mask_query).unsqueeze(0) # [1, D, H, W]
        mask_pos_tensor = torch.from_numpy(mask_pos).unsqueeze(0)     # [1, D, H, W]
        mask_neg_tensor = torch.from_numpy(mask_neg).unsqueeze(0)     # [1, D, H, W]

        # 正负样本的标签，这里表示 (query, pos) 是正对，(query, neg) 是负对
        label_pos_pair = torch.tensor(1.0) # Query-Pos Pair is Positive
        label_neg_pair = torch.tensor(0.0) # Query-Neg Pair is Negative
        
        # 返回原始图像、各个掩码以及对应的标签
        return img_tensor, mask_query_tensor, mask_pos_tensor, mask_neg_tensor, label_pos_pair, label_neg_pair
    
    def _get_mask(self, seg: np.ndarray, mask_id: int) -> np.ndarray:
        """根据mask_id提取二值掩码"""
        mask = np.zeros_like(seg, dtype=np.float32)
        mask[seg == mask_id] = 1
        return mask

# ====================== Squeeze-and-Excitation (SE) Block (保持不变) ======================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

# ====================== U-Net Conv Block (Encoder & Decoder) (保持不变) ======================
class ConvBlock3D(nn.Module):
    """
    3D卷积块，包含两个Conv3d -> BatchNorm3d -> ReLU，并可选择性地包含SEBlock
    """
    def __init__(self, in_channels, out_channels, use_se=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.se_block = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        x = self.block(x)
        x = self.se_block(x)
        return x

# ====================== EmbedNet (U-Net Architecture with SE) (保持不变) ======================
class EmbedNetUNet(nn.Module):
    """
    基于U-Net架构的EmbedNet，包含Squeeze-and-Excitation层。
    输出体素级嵌入。
    """
    def __init__(self, in_channels=1, embed_dim=16, base_channels=32, use_se=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # 编码器 (下采样路径)
        self.enc1 = ConvBlock3D(in_channels, base_channels, use_se) # (128,128,128) -> (128,128,128)
        self.pool1 = nn.MaxPool3d(2) # (128,128,128) -> (64,64,64)

        self.enc2 = ConvBlock3D(base_channels, base_channels * 2, use_se) # (64,64,64) -> (64,64,64)
        self.pool2 = nn.MaxPool3d(2) # (64,64,64) -> (32,32,32)

        self.enc3 = ConvBlock3D(base_channels * 2, base_channels * 4, use_se) # (32,32,32) -> (32,32,32)
        self.pool3 = nn.MaxPool3d(2) # (32,32,32) -> (16,16,16)

        self.enc4 = ConvBlock3D(base_channels * 4, base_channels * 8, use_se) # (16,16,16) -> (16,16,16)
        self.pool4 = nn.MaxPool3d(2) # (16,16,16) -> (8,8,8)

        # 底部 (Bottleneck)
        self.bottleneck = ConvBlock3D(base_channels * 8, base_channels * 16, use_se) # (8,8,8) -> (8,8,8)

        # 解码器 (上采样路径)
        self.upconv4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock3D(base_channels * 16, base_channels * 8, use_se) # (base_channels*8 from upconv + base_channels*8 from enc4)

        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(base_channels * 8, base_channels * 4, use_se)

        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_channels * 4, base_channels * 2, use_se)

        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_channels * 2, base_channels, use_se)

        # 输出层，将通道数映射到 embed_dim
        self.out_conv = nn.Conv3d(base_channels, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码器
        e1 = self.enc1(x)
        p1 = self.pool1(e1) # 64

        e2 = self.enc2(p1)
        p2 = self.pool2(e2) # 32

        e3 = self.enc3(p2)
        p3 = self.pool3(e3) # 16

        e4 = self.enc4(p3)
        p4 = self.pool4(e4) # 8

        # 底部
        b = self.bottleneck(p4) # 8

        # 解码器
        d4 = self.upconv4(b) # 16
        # 如果尺寸不完全匹配，进行裁剪
        d4 = self._center_crop_and_concat(e4, d4) # Concat e4 (16) with d4 (16)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4) # 32
        d3 = self._center_crop_and_concat(e3, d3)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3) # 64
        d2 = self._center_crop_and_concat(e2, d2)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2) # 128
        d1 = self._center_crop_and_concat(e1, d1)
        d1 = self.dec1(d1)

        # 输出体素级嵌入
        return self.out_conv(d1)

    def _center_crop_and_concat(self, enc_feature, dec_feature):
        """
        对编码器特征图进行中心裁剪，以匹配解码器特征图的尺寸，然后拼接。
        """
        enc_shape = enc_feature.shape[2:]
        dec_shape = dec_feature.shape[2:]

        diff_d = enc_shape[0] - dec_shape[0]
        diff_h = enc_shape[1] - dec_shape[1]
        diff_w = enc_shape[2] - dec_shape[2]

        if diff_d < 0 or diff_h < 0 or diff_w < 0:
            # Should not happen in typical U-Net, but for robustness
            raise ValueError("Decoder feature map is larger than encoder feature map for concatenation.")

        crop_d = diff_d // 2
        crop_h = diff_h // 2
        crop_w = diff_w // 2

        enc_feature_cropped = enc_feature[:, :,
                                          crop_d : enc_shape[0] - diff_d + crop_d,
                                          crop_h : enc_shape[1] - diff_h + crop_h,
                                          crop_w : enc_shape[2] - diff_w + crop_w]
        
        # 拼接跳跃连接和上采样特征
        return torch.cat([enc_feature_cropped, dec_feature], dim=1)


# ====================== 对比损失函数 (保持不变，已根据前一次对话修改) ======================
class ContrastiveLoss(nn.Module):
    """对比损失函数 - 处理正负样本对"""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.distance = nn.PairwiseDistance(p=2) # 欧氏距离

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        计算对比损失
        emb1, emb2: 两个嵌入向量 (分段平均嵌入), [B, embed_dim]
        label: 标签 (1表示相似，0表示不相似), [B]
        """
        distance = self.distance(emb1, emb2)

        # 典型的对比损失形式：
        # L = y * D^2 + (1-y) * max(0, margin - D)^2
        # 其中 y=1 表示相似（正样本对），y=0 表示不相似（负样本对）
        # D 是欧氏距离

        # 对于正样本 (label == 1)，希望距离小，损失是距离的平方
        loss_positive = label * torch.pow(distance, 2)
        
        # 对于负样本 (label == 0)，希望距离大，只有当距离小于 margin 时才产生损失
        # Clamp ensures loss is 0 if distance is already larger than margin
        loss_negative = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)

        loss = torch.mean(loss_positive + loss_negative)
        return loss

# ====================== 训练器类 (更新数据集参数传递) ======================
class NeuronTrainer:
    """神经元连接模型训练器"""
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_val_loss = float('inf')
        self.current_val_loss = float('inf')
        self.early_stop_counter = 0  # 早停计数器
        self.patience = config.get('patience', 10)  # 早停容忍epoch数，默认10
        
        # 创建日志目录
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(config['log_dir'], f"train_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(self.log_dir)
        
        # 设置随机种子
        self._set_seed(config['seed'])
        
        # 创建数据集和模型
        self._create_datasets()
        self.model = self._create_model()
        
        # 创建优化器和损失函数
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        self.criterion = ContrastiveLoss(margin=config['margin'])
        
        print(f"训练器初始化完成，设备: {self.device}")

    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _create_model(self) -> nn.Module:
        """创建并配置模型"""
        # 实例化EmbedNetUNet
        model = EmbedNetUNet(
            in_channels=1, 
            embed_dim=self.config['embed_dim'],
            base_channels=self.config.get('base_channels', 32), # 可以在config中指定base_channels
            use_se=self.config.get('use_se', True)
        ).to(self.device)
        
        # 打印模型信息
        print(f"模型架构:")
        print(model)
        print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    def _create_datasets(self):
        """创建训练和验证数据集"""
        # 训练数据集
        self.train_dataset = NeuronConnectivityDataset(
            data_root=self.config['data_root'],
            img_dir=self.config['img_dir'],
            seg_dir=self.config['seg_dir'],
            split_file=self.config['train_split'],
            volume_size=self.config['volume_size'],
            augment=self.config.get('augment', False), # 从 config 获取 augment 参数
            max_translation=self.config.get('max_translation', 10) # 从 config 获取 max_translation 参数
        )
        
        # 验证数据集 (通常验证集不进行数据增强)
        self.val_dataset = NeuronConnectivityDataset(
            data_root=self.config['data_root'],
            img_dir=self.config['img_dir'],
            seg_dir=self.config['seg_dir'],
            split_file=self.config['val_split'],
            volume_size=self.config['volume_size'],
            augment=False, # 验证集通常不增强
            max_translation=0 # 验证集不增强，所以平移量为0
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        print(f"训练集大小: {len(self.train_dataset)} 个体积")
        print(f"验证集大小: {len(self.val_dataset)} 个体积")

    def _combine_inputs(self, img_batch: torch.Tensor, 
                         mask_query_batch: torch.Tensor,
                         mask_pos_batch: torch.Tensor,
                         mask_neg_batch: torch.Tensor) -> torch.Tensor:
        """
        组合输入：图像 + 3个掩码，作为EmbedNet的输入
        """
        # img_batch: [B, 1, D, H, W]
        # mask_X_batch: [B, 1, D, H, W]
        
        # 拼接图像和掩码 [B, 4, D, H, W]
        combined = torch.cat([img_batch, mask_query_batch, mask_pos_batch, mask_neg_batch], dim=1)
        
        return combined

    def _get_segment_embeddings(self, voxel_embeddings: torch.Tensor, 
                                 mask: torch.Tensor) -> torch.Tensor:
        """
        根据体素级嵌入和对应分段的掩码计算分段平均嵌入。
        
        voxel_embeddings: [B, embed_dim, D', H', W'] - EmbedNet的输出 (例如 8x8x8)
        mask: [B, 1, D, H, W] - 原始尺寸的二值掩码 (例如 128x128x128)
        
        返回: [B, embed_dim] - 分段的平均嵌入
        """
        # 获取体素嵌入的空间维度
        _, _, D_prime, H_prime, W_prime = voxel_embeddings.shape
        
        # 将掩码下采样到与体素嵌入相同的空间维度
        downsampled_mask = F.interpolate(
            mask, 
            size=(D_prime, H_prime, W_prime), 
            mode='nearest' 
        )
        
        # 确保掩码是二值的
        downsampled_mask = (downsampled_mask > 0.5).float()

        # 扩展掩码维度以匹配embed_dim，以便进行元素乘法
        # [B, 1, D', H', W'] -> [B, embed_dim, D', H', W']
        expanded_mask = downsampled_mask.expand_as(voxel_embeddings)
        
        # 提取分段内的体素嵌入
        segment_voxel_features = voxel_embeddings * expanded_mask
        
        # 计算每个分段中非零（即属于分段）体素的数量
        # [B, 1, D', H', W'] -> [B]
        num_voxels = torch.sum(downsampled_mask.squeeze(1), dim=(-3, -2, -1))
        
        # 计算分段平均嵌入
        # 对空间维度求和 [B, embed_dim, D', H', W'] -> [B, embed_dim]
        sum_features = torch.sum(segment_voxel_features, dim=(-3, -2, -1))
        
        # 避免除以零，对于空分段（num_voxels=0），嵌入可以设为零向量
        avg_embeddings = torch.where(
            num_voxels.unsqueeze(1) > 0, 
            sum_features / num_voxels.unsqueeze(1),
            torch.zeros_like(sum_features) 
        )
        
        return avg_embeddings

    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"训练 Epoch {epoch}")
        
        # (img_tensor, mask_query_tensor, mask_pos_tensor, mask_neg_tensor, label_pos_pair, label_neg_pair)
        for batch_idx, (imgs, mask_q, mask_p, mask_n, label_pp, label_np) in enumerate(progress_bar):
            # 移动到设备
            imgs, mask_q, mask_p, mask_n = imgs.to(self.device), mask_q.to(self.device), \
                                             mask_p.to(self.device), mask_n.to(self.device)
            label_pp, label_np = label_pp.to(self.device), label_np.to(self.device)
            
            
            self.optimizer.zero_grad()
            
            # 前向传播，获取体素级嵌入特征图
            # voxel_embeddings: [B, embed_dim, D', H', W']
            voxel_embeddings = self.model(imgs)
            
            # 根据论文，从体素级嵌入中计算分段的平均嵌入
            # 这里计算的是 Query Segment (e_Q), Positive Segment (e_P), Negative Segment (e_N) 的平均嵌入
            # mask_q, mask_p, mask_n 是原始尺寸的掩码
            e_query = self._get_segment_embeddings(voxel_embeddings, mask_q)
            e_pos = self._get_segment_embeddings(voxel_embeddings, mask_p)
            e_neg = self._get_segment_embeddings(voxel_embeddings, mask_n)
            # 计算对比损失
            # 1. 促使 (e_query, e_pos) 相似 (label_pp = 1.0)
            loss_qp = self.criterion(e_query, e_pos, label_pp) 
            
            # 2. 促使 (e_query, e_neg) 不相似 (label_np = 0.0)
            loss_qn = self.criterion(e_query, e_neg, label_np) 
            
            # 总损失
            loss = loss_qp + loss_qn
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            
            # 更新进度条
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix(loss=f"{loss.item():.16f}", avg_loss=f"{avg_loss:.16f}", loss_qp=f"{loss_qp.item():.16f}", loss_qn=f"{loss_qn.item():.16f}")
            
            # 记录TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Loss/train_batch', loss.item(), global_step)
        
        return total_loss / len(self.train_loader)
    
    def validate(self, epoch: int) -> float:
        """在验证集上评估"""
        self.model.eval()
        total_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc=f"验证 Epoch {epoch}")
        
        with torch.no_grad():
            # (img_tensor, mask_query_tensor, mask_pos_tensor, mask_neg_tensor, label_pos_pair, label_neg_pair)
            for batch_idx, (imgs, mask_q, mask_p, mask_n, label_pp, label_np) in enumerate(progress_bar):
                imgs, mask_q, mask_p, mask_n = imgs.to(self.device), mask_q.to(self.device), \
                                                 mask_p.to(self.device), mask_n.to(self.device)
                label_pp, label_np = label_pp.to(self.device), label_np.to(self.device)
                
                
                # 前向传播
                voxel_embeddings = self.model(imgs)
                
                # 计算分段平均嵌入
                e_query = self._get_segment_embeddings(voxel_embeddings, mask_q)
                e_pos = self._get_segment_embeddings(voxel_embeddings, mask_p)
                e_neg = self._get_segment_embeddings(voxel_embeddings, mask_n)
                
                # 计算损失
                loss_qp = self.criterion(e_query, e_pos, label_pp)
                loss_qn = self.criterion(e_query, e_neg, label_np)
                loss = loss_qp + loss_qn
                total_loss += loss.item()
                
                # 更新进度条
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")
        
        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存模型检查点"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.current_val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.log_dir, 'checkpoint_latest.pth')
        torch.save(state, latest_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(self.log_dir, 'checkpoint_best.pth')
            torch.save(state, best_path)
            print(f"🔥 保存最佳模型: 验证损失 {self.best_val_loss:.4f}")
    
    def train(self):
        """主训练循环"""
        print("\n" + "="*60)
        print(f"开始训练 {self.config['epochs']} 个周期")
        print("="*60)
        print(f"日志目录: {self.log_dir}")
        print(f"输入尺寸: {self.config['volume_size']}")
        
        for epoch in range(1, self.config['epochs'] + 1):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            self.current_val_loss = self.validate(epoch)
            
            # 检查是否为最佳模型
            is_best = self.current_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = self.current_val_loss
                self.early_stop_counter = 0  # 重置早停计数器
            else:
                self.early_stop_counter += 1
                print(f"   早停计数: {self.early_stop_counter}/{self.patience}")
            
            # 保存检查点
            self.save_checkpoint(epoch, is_best)
            
            # 打印epoch摘要
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch}/{self.config['epochs']} 摘要:")
            print(f"   训练损失: {train_loss:.4f} | 验证损失: {self.current_val_loss:.4f}")
            print(f"   时间: {epoch_time:.1f}秒 | 学习率: {self.config['lr']:.6f}")
            
            if is_best:
                print(f"   🎯 新的最佳验证损失: {self.best_val_loss:.4f}")
            
            # 显式释放内存
            torch.cuda.empty_cache()
            gc.collect()
            
            # 早停判断
            if self.early_stop_counter >= self.patience:
                print(f"\n>>> 早停触发: 验证损失连续 {self.patience} 个epoch未提升，提前终止训练。")
                break
        
        print("\n" + "="*60)
        print(f"训练完成! 最佳验证损失: {self.best_val_loss:.4f}")
        print("="*60)
        self.writer.close()

# ====================== 主函数 ======================
if __name__ == "__main__":
    # 训练配置
    config = {
        # 数据配置
        'data_root': '/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/data',
        'img_dir': 'img',
        'seg_dir': 'seg',
        'train_split': 'train.txt',
        'val_split': 'val.txt',
        'volume_size': (128, 128, 128),  # 固定体积尺寸
        'augment': True,                 # 启用数据增强
        'max_translation': 10,           # 最大平移像素值 (例如，在D,H,W方向上可平移-10到+10像素)
        
        # 模型配置
        'embed_dim': 16, # 每个体素的嵌入维度
        'base_channels': 32, # U-Net初始通道数
        'use_se': True, # 是否使用SE层

        # 训练配置
        'batch_size': 8,
        'epochs': 50,
        'lr': 0.001,
        'num_workers': 4,
        'seed': 42,
        
        # 损失配置
        'margin': 10,
        
        # 日志配置
        'log_dir': './logs',
        'patience': 5,  # 新增：早停容忍epoch数
    }
    
    # 创建并运行训练器
    trainer = NeuronTrainer(config)
    trainer.train()