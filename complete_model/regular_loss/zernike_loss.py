import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
import joblib
import sys
sys.path.append('/nvme2/mingzhi/NucCorr')
# 假设 Zernike.py 中的 compute_zernike_descriptor_from_tensor 已经是用 PyTorch 实现的可微分函数
# 请确保你的实际 Zernike 实现是可微分的！
from NucDet.Zernike import compute_zernike_descriptor_from_tensor

class DifferentiableGMM(nn.Module):
    def __init__(self, n_components, n_features, means, covariances, weights, device='cuda'):
        super(DifferentiableGMM, self).__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.device = device
        self.register_buffer('weights_pyt', torch.tensor(weights, dtype=torch.float64, device=device))
        self.register_buffer('means_pyt', torch.tensor(means, dtype=torch.float64, device=device))
    
        precisions_chol = []
        for i in range(n_components):
            cov = torch.tensor(covariances[i], dtype=torch.float64, device=device)
            cov = torch.diag(cov)
            # 为了数值稳定性，对协方差矩阵加上一个小的对角扰动
            cov = cov + torch.eye(self.n_features, device=device) * 1e-6
            # 这里的逻辑是获取协方差矩阵的 Cholesky 分解的下三角因子 (L)
            L = torch.linalg.cholesky(cov) 
            precisions_chol.append(L)
        
        self.register_buffer('precisions_chol_pyt', torch.stack(precisions_chol))

    def forward(self, x):
        log_prob_components = []
        
        for k in range(self.n_components):
            mean_k = self.means_pyt[k]
            m_k = MultivariateNormal(loc=mean_k, scale_tril=self.precisions_chol_pyt[k])
            log_prob_k = m_k.log_prob(x)
            log_prob_components.append(log_prob_k + torch.log(self.weights_pyt[k]))
        log_likelihood = torch.logsumexp(torch.stack(log_prob_components, dim=1), dim=1)
        
        return log_likelihood

    def negative_log_likelihood(self, X):
        return -self.forward(X).mean()
        
    