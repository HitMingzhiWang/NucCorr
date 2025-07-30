import os
import torch
import numpy as np
import tifffile
from pathlib import Path
from model.unet import get_model
from torch.utils.data import Dataset, DataLoader

class NucleiDataset(Dataset):
    def __init__(self, dataset_dir, file_list, drop_type=-1):
        self.dataset_dir = dataset_dir
        self.file_list = file_list
        self.drop_type = drop_type
        
    def __len__(self):
        return len(self.file_list)
    
    def apply_drop(self, seg, img):
        """应用不同的drop策略"""
        if self.drop_type == -1:  # 不drop
            return seg, img
            
        h, w = seg.shape
        if self.drop_type == 0:  # patch drop
            # 随机选择一个patch区域
            patch_size = min(h, w) // 4
            x = np.random.randint(0, w - patch_size)
            y = np.random.randint(0, h - patch_size)
            seg[y:y+patch_size, x:x+patch_size] = 0
            
        elif self.drop_type == 1:  # slice drop from start
            # 从开始位置drop一个slice
            slice_width = w // 4
            seg[:, :slice_width] = 0
            
        elif self.drop_type == 2:  # slice drop from middle
            # 从中间位置drop一个slice
            slice_width = w // 4
            start = (w - slice_width) // 2
            seg[:, start:start+slice_width] = 0
            
        elif self.drop_type == 3:  # slice drop from end
            # 从结束位置drop一个slice
            slice_width = w // 4
            seg[:, -slice_width:] = 0
            
        return seg, img
    
    def __getitem__(self, idx):
        file_id = self.file_list[idx]
        seg_path = os.path.join(self.dataset_dir, "seg", f"{file_id}.tiff")
        img_path = os.path.join(self.dataset_dir, "img", f"{file_id}.tiff")
        
        # 读取数据
        img = tifffile.imread(img_path)
        seg = tifffile.imread(seg_path)
        
        # 应用drop策略
        seg, img = self.apply_drop(seg, img)
        
        # 保存drop后的输入
        debug_dir = os.path.join("val_test_results", 'debug', file_id)
        os.makedirs(debug_dir, exist_ok=True)
        tifffile.imwrite(os.path.join(debug_dir, 'input_seg.tiff'), seg)
        tifffile.imwrite(os.path.join(debug_dir, 'input_img.tiff'), img)
        
        # 转换为tensor
        seg = seg.astype(np.uint8)
        seg = torch.from_numpy(seg).float().unsqueeze(0)
        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        
        return {
            'file_id': file_id,
            'img': img,
            'seg': seg
        }

def process_batch(batch, model, device, output_dir):
    """处理一个batch的数据"""
    file_ids = batch['file_id']
    imgs = batch['img'].to(device)
    segs = batch['seg'].to(device)
    
    # 模型预测
    with torch.no_grad():
        x = torch.cat([segs, imgs], dim=1)
        output = model(x)
        preds = (torch.sigmoid(output) > 0.3).float()
    
    # 保存结果
    for i, file_id in enumerate(file_ids):
        # 保存预测结果
        pred_path = os.path.join(output_dir, 'pred', f'{file_id}_pred.tiff')
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        tifffile.imwrite(pred_path, preds[i].cpu().numpy().astype(np.uint8))
        print(f"Successfully processed {file_id}")

def main():
    # 设置路径
    dataset_dir = "/nvme2/mingzhi/NucCorr/NucCorrData/normal_nuclei/normal_nuclei"
    model_path = "/nvme2/mingzhi/NucCorr/complete_model/checkpoints/best_model_epoch_32.pth"
    output_dir = "val_test_results"
    
    # 设置drop类型
    drop_type = 0  # -1: no drop, 0: patch drop, 1: slice drop from start, 2: slice drop from middle, 3: slice drop from end
    
    # 读取验证集文件列表
    val_list_path = os.path.join(dataset_dir, "normal_nuclei_val.txt")
    with open(val_list_path, 'r') as f:
        val_files = [line.strip() for line in f.readlines()]
    
    # 选择前20个文件进行测试
    test_files = val_files[:20]
    print(f"Selected {len(test_files)} files for testing")
    print(f"Using drop type: {drop_type}")
    
    # 设置设备
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建数据集和数据加载器
    dataset = NucleiDataset(dataset_dir, test_files, drop_type=drop_type)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    
    # 加载模型
    model = get_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 处理数据
    successful = 0
    for batch in dataloader:
        try:
            process_batch(batch, model, device, output_dir)
            successful += len(batch['file_id'])
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {successful}/{len(test_files)} files")

if __name__ == "__main__":
    main() 