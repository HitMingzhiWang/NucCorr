import os
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from neuron_dataset import PointCloudDataset

if __name__ == '__main__':
    img_dir = '/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/data/img'
    seg_dir = '/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/data/seg'
    annotation_file = '/nvme2/mingzhi/NucCorr/correct_baseline/MIDL/data/train.txt'
    dataset = PointCloudDataset(
        img_dir=img_dir,
        seg_dir=seg_dir,
        annotation_file=annotation_file,
        num_points=2048,
        is_training=False,
        normalize=True
    )
    print(f"Dataset size: {len(dataset)}")
    points, label = dataset[6000]
    print(f"Point cloud shape: {points.shape}")
    print(f"Label: {label}")

    # 三维可视化
    z = points[:, 0].numpy()
    y = points[:, 1].numpy()
    x = points[:, 2].numpy()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Point Cloud (zxy)')
    fig.colorbar(sc, label='z')
    plt.tight_layout()
    plt.savefig('/nvme2/mingzhi/NucCorr/correct_baseline/MIDL/dataset/pointcloud_3d_vis.png')
    plt.show()