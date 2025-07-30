import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import seaborn as sns
import pickle
import umap  # 导入UMAP库
import os
# Step 1: 读取数据
npz_path = '/nvme2/mingzhi/NucCorr/NucDet/features_split_centered.npz'
data = np.load(npz_path)

features_list = []
keys_list = []

for key in data.files:
    feature = data[key]
    if feature.shape[-1] != 121:  # 检查特征维度
        print(f"跳过ID {key}，特征维度不正确: {feature.shape}")
        continue
    features_list.append(feature)
    keys_list.append(key)

# 转换为numpy数组
features = np.array(features_list)
keys = np.array(keys_list)

print(f"Features shape: {features.shape}")
print(f"Keys shape: {keys.shape}")
print(f"Number of samples: {len(features)}")
print(f"Sample keys: {keys[:5]}")

# 检查数据质量
print(f"Features range: [{features.min():.6f}, {features.max():.6f}]")
print(f"Features mean: {features.mean():.6f}")
print(f"Features std: {features.std():.6f}")
print(f"Any NaN values: {np.any(np.isnan(features))}")
print(f"Any Inf values: {np.any(np.isinf(features))}")

# Step 2: GMM 拟合（使用原始特征）
gmm = GaussianMixture(n_components=1, covariance_type='diag', random_state=42)
gmm.fit(features)
labels = gmm.predict(features)

# 打印聚类结果统计
print(f"\nUnique labels: {np.unique(labels)}")
print(f"Number of unique clusters: {len(np.unique(labels))}")
for cluster_id in np.unique(labels):
    cluster_size = np.sum(labels == cluster_id)
    print(f"Cluster {cluster_id}: {cluster_size} samples ({cluster_size/len(labels)*100:.1f}%)")

# Step 3: 使用UMAP降维（替代PCA）
# 创建UMAP降维器
reducer = umap.UMAP(
    n_components=2,          # 降到2维
    n_neighbors=15,          # 考虑局部邻域的大小
    min_dist=0.1,            # 控制点的聚集程度
    metric='euclidean',      # 距离度量方式
    random_state=42          # 随机种子
)

# 执行降维（使用原始特征）
features_2d = reducer.fit_transform(features)

print(f"\nUMAP 2D features range: [{features_2d.min():.6f}, {features_2d.max():.6f}]")

# Step 4: 可视化聚类结果
plt.figure(figsize=(12, 10))

# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# 第一个子图：所有点的散点图（不按聚类着色）
ax1.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6, s=20, color='blue')
ax1.set_title(f"All {len(features)} points (no clustering)")
ax1.set_xlabel("UMAP-1")
ax1.set_ylabel("UMAP-2")
ax1.grid(True)

# 第二个子图：按聚类着色的散点图
palette = sns.color_palette("hsv", n_colors=np.max(labels)+1)
total_points_plotted = 0
for cluster_id in np.unique(labels):
    idx = labels == cluster_id
    cluster_points = features_2d[idx]
    cluster_size = np.sum(idx)
    total_points_plotted += cluster_size
    
    print(f"Plotting cluster {cluster_id}: {cluster_size} points")
    ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                label=f"Cluster {cluster_id} ({cluster_size} points)", 
                alpha=0.7, s=30, color=palette[cluster_id])

print(f"Total points plotted in second subplot: {total_points_plotted}")

ax2.set_title("GMM Clustering of 3D Zernike Descriptors (UMAP Projection)")
ax2.set_xlabel("UMAP-1")
ax2.set_ylabel("UMAP-2")
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True)

plt.tight_layout()
plt.savefig("gmm_zernike_clustering_umap.png", dpi=300, bbox_inches='tight')

# 额外创建一个简单的散点图来验证所有点
plt.figure(figsize=(10, 8))
plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.5, s=10, c=labels, cmap='tab10')
plt.colorbar(label='Cluster ID')
plt.title(f"All {len(features_2d)} points colored by cluster (UMAP)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.savefig("simple_clustering_verification_umap.png", dpi=300, bbox_inches='tight')
plt.show()

# 保存GMM模型
with open('/nvme2/mingzhi/NucCorr/NucDet/gmm_zernike_model.pkl', 'wb') as f:
    pickle.dump(gmm, f)
print("GMM model saved to gmm_zernike_model.pkl")

# 额外的调试信息
print(f"\nTotal points plotted: {len(features_2d)}")
print(f"Expected total points: {len(features)}")
print(f"Points per cluster:")
for cluster_id in np.unique(labels):
    cluster_size = np.sum(labels == cluster_id)
    print(f"  Cluster {cluster_id}: {cluster_size} points")

out_dir = '/nvme2/mingzhi/NucCorr/NucDet/gmm_clusters_txt'
os.makedirs(out_dir, exist_ok=True)

# keys: numpy array of sample keys, labels: numpy array of cluster ids
for cluster_id in np.unique(labels):
    idx = labels == cluster_id
    cluster_keys = keys[idx]
    out_path = os.path.join(out_dir, f'cluster_{cluster_id}.txt')
    with open(out_path, 'w') as f:
        for k in cluster_keys:
            f.write(f"{k}\n")
    print(f"Cluster {cluster_id}: {len(cluster_keys)} keys written to {out_path}")