import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import umap

def load_features_with_labels(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    features = data['features']
    keys = data['keys']
    labels = data['labels']
    return features, keys, labels

def handle_nan_values(features, keys, labels, strategy='remove'):
    nan_mask = np.isnan(features).any(axis=1)
    nan_count = nan_mask.sum()
    total_samples = features.shape[0]
    print(f"检测到 {nan_count}/{total_samples} 个样本包含NaN值")
    if nan_count == 0:
        return features, keys, labels
    if strategy == 'remove':
        clean_features = features[~nan_mask]
        clean_keys = [key for i, key in enumerate(keys) if not nan_mask[i]]
        clean_labels = [label for i, label in enumerate(labels) if not nan_mask[i]]
        print(f"已删除 {nan_count} 个包含NaN值的样本")
        return clean_features, clean_keys, clean_labels
    elif strategy == 'mean':
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        clean_features = imputer.fit_transform(features)
        print(f"已使用特征均值填充 {nan_count} 个NaN值")
        return clean_features, keys, labels
    else:
        raise ValueError(f"未知策略: {strategy}。请选择 'remove' 或 'mean'")

def visualize_with_umap_label(features, labels, random_state=42):
    from matplotlib.colors import ListedColormap
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    print("开始UMAP降维...")
    umap_results = reducer.fit_transform(features_scaled)
    print("UMAP降维完成")

    # 固定label顺序
    label_order = ['normal', 'split', 'merge']
    present_labels = [lab for lab in label_order if lab in np.unique(labels)]
    label_to_color = {lab: idx for idx, lab in enumerate(present_labels)}
    color_indices = np.array([label_to_color[lab] for lab in labels])

    cmap = ListedColormap(plt.cm.tab10.colors[:len(present_labels)])

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(umap_results[:, 0], umap_results[:, 1], c=color_indices, cmap=cmap, alpha=0.7, s=50)
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=lab,
                          markerfacecolor=cmap(idx), markersize=10)
               for idx, lab in enumerate(present_labels)]
    plt.legend(handles=handles, title='Label', loc='best')
    plt.title(f'UMAP Visualization of 3D Zernike Features by Label\n({features.shape[0]} samples, {features.shape[1]} features)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.grid(alpha=0.3)
    output_png = 'umap_visualization_by_label.png'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"可视化结果已保存至: {os.path.abspath(output_png)}")
    return umap_results

if __name__ == "__main__":
    npz_path = "features_with_label.npz"  # 替换为你的带标签NPZ文件路径
    nan_strategy = 'remove'

    # 加载特征数据
    features, keys, labels = load_features_with_labels(npz_path)
    print(f"已加载特征数据: {features.shape[0]}个样本, {features.shape[1]}维特征")
    # 处理NaN值
    features_clean, keys_clean, labels_clean = handle_nan_values(features, keys, labels, strategy=nan_strategy)

    # ====== 随机选3000个normal，全部split，全部merge ======
    features_clean = np.array(features_clean)
    keys_clean = np.array(keys_clean)
    labels_clean = np.array(labels_clean)

    normal_idx = np.where(labels_clean == 'normal')[0]
    split_idx = np.where(labels_clean == 'split')[0]
    merge_idx = np.where(labels_clean == 'merge')[0]

    # 随机选3000个normal
    if len(normal_idx) > 29000:
        np.random.seed(618)
        normal_idx = np.random.choice(normal_idx, 29000, replace=False)

    selected_idx = np.concatenate([normal_idx, split_idx, merge_idx])
    features_selected = features_clean[selected_idx]
    labels_selected = labels_clean[selected_idx]
    keys_selected = keys_clean[selected_idx]

    if features_selected.size == 0:
        print("错误: 所有样本都包含NaN值！无法进行可视化")
    else:
        print(f"处理后数据: {features_selected.shape[0]}个样本, {features_selected.shape[1]}维特征")
        umap_results = visualize_with_umap_label(features_selected, labels_selected)