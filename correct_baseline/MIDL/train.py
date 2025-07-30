import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import random
import numpy as np
from dataset.neuron_dataset import PointCloudDataset
from model.point_net import PointNet2Classification

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

train_txt = '/nvme2/mingzhi/NucCorr/correct_baseline/MIDL/data/train.txt'
val_txt = '/nvme2/mingzhi/NucCorr/correct_baseline/MIDL/data/val.txt'
img_dir = '/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/data/img'
seg_dir = '/nvme2/mingzhi/NucCorr/correct_baseline/contrastive_learning/data/seg'
batch_size = 32
num_workers = 4
num_epochs = 50
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "best_pointnet.pth"
save_every = 10
num_points = 2048

train_dataset = PointCloudDataset(
    img_dir=img_dir,
    seg_dir=seg_dir,
    annotation_file=train_txt,
    num_points=num_points,
    is_training=True,
    normalize=True
)
val_dataset = PointCloudDataset(
    img_dir=img_dir,
    seg_dir=seg_dir,
    annotation_file=val_txt,
    num_points=num_points,
    is_training=False,
    normalize=True
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

model = PointNet2Classification(num_classes=1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
best_f1 = 0.0

def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch_idx, (points, labels) in enumerate(pbar):
        points, labels = points.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(points)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = (outputs > 0.5).int().squeeze(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=loss.item(), batch=batch_idx)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    print(f"✅ Train Epoch {epoch}: Loss={total_loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

def validate(epoch):
    global best_f1
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
    with torch.no_grad():
        for batch_idx, (points, labels) in enumerate(pbar):
            points, labels = points.to(device), labels.to(device)
            outputs = model(points)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            total_loss += loss.item()
            preds = (outputs > 0.5).int().squeeze(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix(loss=loss.item(), batch=batch_idx)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    print(f"📊 Val Epoch {epoch}: Loss={total_loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Saved best checkpoint at epoch {epoch} with F1={f1:.4f}")

if __name__ == '__main__':
    set_seed(42)
    print(f"Using device: {device}")
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    for epoch in range(1, num_epochs + 1):
        train_one_epoch(epoch)
        validate(epoch)
        if epoch % save_every == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")
            print(f"💾 Saved checkpoint for epoch {epoch}.")