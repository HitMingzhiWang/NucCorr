import torch
import numpy as np
import joblib
from Zernike import compute_zernike_descriptor_from_tensor
from skimage import io


# 1. 加载GMM模型
gmm = joblib.load("/nvme2/mingzhi/NucCorr/NucDet/gmm_zernike_model.pkl")

obj = io.imread('/nvme2/mingzhi/NucCorr/NucDet/pred.tiff')
obj = np.where(obj ==1 ,1 , 0).astype(np.float32)
obj_tensor = torch.from_numpy(obj).to(torch.float64).cuda()  # 或cpu
zernike_feature = compute_zernike_descriptor_from_tensor(obj_tensor, max_order=20, device='cuda')
zernike_feature = zernike_feature.detach().cpu().numpy().reshape(1, -1)  # shape (1, 121)

print(f"特征形状: {zernike_feature.shape}")
print(f"特征范围: [{zernike_feature.min():.6f}, {zernike_feature.max():.6f}]")
print(f"特征均值: {zernike_feature.mean():.6f}")
print(f"特征标准差: {zernike_feature.std():.6f}")

print(f"GMM模型均值形状: {gmm.means_.shape}")

# 2. 计算负对数似然（使用原始特征）
nll = -gmm.score_samples(zernike_feature)[0]
print("负对数似然(NLL):", nll)
val = gmm.score_samples(zernike_feature)[0]
print(f"Score: {val:.6f}, exp(score): {np.exp(val):.6f}")