import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import json
import os
from collections import defaultdict
from config import OUTPUT_DIR
from st_gcn_model import STGCN
import random

class STGCNDataset(Dataset):
    def __init__(self, data_path, labels_path):
        self.data = torch.FloatTensor(np.load(data_path))
        self.labels = torch.LongTensor(np.load(labels_path))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        # If loss improves
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

def train_model(batch_size=16, epochs=50, lr=0.001, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Training on {device}")

    # Load data
    data_path = os.path.join(OUTPUT_DIR, 'stgcn_data_aug.npy')
    labels_path = os.path.join(OUTPUT_DIR, 'stgcn_labels_aug.npy')
    dataset = STGCNDataset(data_path, labels_path)
    labels = np.load(labels_path)

    # Split dataset theo label
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    train_indices = []
    val_indices = []
    for label, idxs in label_to_indices.items():
        random.shuffle(idxs)
        n_train = max(1, int(0.8 * len(idxs)))
        train_indices += idxs[:n_train]
        val_indices += idxs[n_train:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Label map
    with open(os.path.join(OUTPUT_DIR, 'label_map.json'), 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    num_classes = len(label_map)

    # Graph
    A = np.load(os.path.join(OUTPUT_DIR, 'mediapipe_graph.npy'))

    # Model
    model = STGCN(in_channels=3, num_class=num_classes, A=A).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Early Stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=0.0005)

    best_val_acc = 0
    best_model_path = os.path.join(OUTPUT_DIR, "stgcn_best.pth")

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # TRAIN
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total
        train_loss = total_loss / len(train_loader)

        # VALIDATION
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss_total += loss.item()
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss_total / len(val_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )

        # Save best model by val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved Best Model (Val Acc = {best_val_acc:.2f}%)")

        # Early Stopping check
        if early_stopping(val_loss):
            print("Early stopping triggered")
            break

    print("🔥 Training finished!")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")

    # Save model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'stgcn_mediapipe75.pth'))
    print("Model saved")

if __name__=="__main__":
    train_model()
