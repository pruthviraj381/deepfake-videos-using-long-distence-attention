"""
Retrain Long-Distance Attention Deepfake Detector on Real & Manipulated Video Clips
Produces accurate calibration so authentic human videos predict REAL with high confidence,
and deepfake videos with boundary or temporal warping artifacts predict FAKE with high confidence.
"""

import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train import DeepfakeVideoDataset, BEST_MODEL_PATH
from model import create_model
import database

def retrain(epochs=10, batch_size=4, lr=0.0005):
    print("=== Training Long-Distance Attention Detector on Realistic Dataset ===")
    train_dataset = DeepfakeVideoDataset(split='train', num_frames=4, target_size=(64, 64))
    val_dataset = DeepfakeVideoDataset(split='val', num_frames=4, target_size=(64, 64))

    print(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model = create_model(num_frames=4, image_size=64, d_model=64, num_heads=4)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = (correct / max(total, 1)) * 100.0

        # Evaluate on validation set
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                v_correct += torch.sum(preds == labels.data).item()
                v_total += labels.size(0)

        val_loss = v_loss / max(v_total, 1)
        val_acc = (v_correct / max(v_total, 1)) * 100.0

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%")

        if val_acc >= best_val_acc or best_weights is None:
            best_val_acc = val_acc
            best_weights = model.state_dict().copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_weights,
                'val_acc': best_val_acc,
                'num_frames': 4,
                'image_size': 64,
                'd_model': 64,
                'num_heads': 4
            }, BEST_MODEL_PATH)

    print(f"\n[Training Complete] Best Validation Accuracy: {best_val_acc:.1f}% saved to {BEST_MODEL_PATH}")

    # Log into SQLite
    database.init_db()
    database.log_training_run(
        admin_id=1,
        epochs=epochs,
        batch_size=batch_size,
        train_loss=round(train_loss, 4),
        train_accuracy=round(train_acc, 2),
        val_loss=round(val_loss, 4),
        val_accuracy=round(best_val_acc, 2),
        checkpoint_path=BEST_MODEL_PATH,
        real_count=len([s for s in train_dataset.samples if s[1] == 0]),
        fake_count=len([s for s in train_dataset.samples if s[1] == 1]),
        status="completed"
    )

if __name__ == "__main__":
    retrain(epochs=10, batch_size=4, lr=0.0005)
