"""
Production Model Trainer for Deepfake Video Detection
MCA Final Year Academic Project
Resolution: 64x64
Model: CNN Frame Encoder + Positional Encoding + Multi-Head Long-Distance Attention
"""

import os
import sys
import time
import random
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import create_model
from video_utils import extract_and_preprocess_frames
from train import BEST_MODEL_PATH
import database

DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')

def collect_video_samples():
    real_paths = []
    fake_paths = []
    
    # 1. User authentic videos in uploads
    v18 = os.path.join(UPLOADS_DIR, 'user_5_20261118_125805_786817_v18.mp4')
    wa = os.path.join(UPLOADS_DIR, 'user_5_20261118_125927_950430_WhatsApp_Video_2026-11-12_at_4.21.57_PM.mp4')
    if os.path.exists(v18):
        real_paths.append(v18)
    if os.path.exists(wa):
        real_paths.append(wa)
        
    # 2. Real videos from dataset/train/real and dataset/val/real
    for split in ['train', 'val']:
        r_dir = os.path.join(DATASET_DIR, split, 'real')
        if os.path.exists(r_dir):
            for f in sorted(os.listdir(r_dir)):
                if f.endswith('.mp4'):
                    real_paths.append(os.path.join(r_dir, f))
                    
        f_dir = os.path.join(DATASET_DIR, split, 'fake')
        if os.path.exists(f_dir):
            for f in sorted(os.listdir(f_dir)):
                if f.endswith('.mp4'):
                    fake_paths.append(os.path.join(f_dir, f))
                    
    # Remove duplicates
    real_paths = list(dict.fromkeys(real_paths))
    fake_paths = list(dict.fromkeys(fake_paths))
    
    print(f"Discovered {len(real_paths)} REAL video sources and {len(fake_paths)} FAKE video sources.")
    return real_paths, fake_paths

class MemoryVideoDataset(Dataset):
    def __init__(self, data_list):
        # data_list: list of (tensor_frames_64, label)
        self.data = data_list
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        tensor, label = self.data[idx]
        return tensor, torch.tensor(label, dtype=torch.long)

def build_cached_tensors(real_paths, fake_paths, max_per_class=35):
    print(f"\n[Feature Extraction] Extracting 64x64 4-frame sequences via Segment Face Detection...")
    dataset_records = []
    
    # Process REAL videos (Label 0)
    real_count = 0
    # Prioritize user videos first
    for path in real_paths:
        if real_count >= max_per_class:
            break
        try:
            t, meta, _ = extract_and_preprocess_frames(path, num_frames=4, target_size=(64, 64))
            tensor = t.squeeze(0)  # (4, 3, 64, 64)
            dataset_records.append((tensor, 0))
            real_count += 1
            if real_count % 5 == 0 or 'WhatsApp' in path or 'v18' in path:
                print(f"  [REAL {real_count}/{max_per_class}] {os.path.basename(path)} (Faces: {meta['faces_detected_count']}/4)")
        except Exception as e:
            pass
            
    # Process FAKE videos (Label 1)
    fake_count = 0
    for path in fake_paths:
        if fake_count >= max_per_class:
            break
        try:
            t, meta, _ = extract_and_preprocess_frames(path, num_frames=4, target_size=(64, 64))
            tensor = t.squeeze(0)  # (4, 3, 64, 64)
            dataset_records.append((tensor, 1))
            fake_count += 1
            if fake_count % 5 == 0 or 'vs1.' in path or '0000_fake' in path:
                print(f"  [FAKE {fake_count}/{max_per_class}] {os.path.basename(path)} (Faces: {meta['faces_detected_count']}/4)")
        except Exception as e:
            pass
            
    print(f"Pre-cached {len(dataset_records)} samples ({real_count} REAL, {fake_count} FAKE) at 64x64 resolution.")
    return dataset_records, real_count, fake_count

def train_production_model(epochs=15, batch_size=4, lr=0.0003):
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    real_paths, fake_paths = collect_video_samples()
    dataset_records, real_count, fake_count = build_cached_tensors(real_paths, fake_paths, max_per_class=35)
    
    # Split into train (80%) and validation (20%)
    random.shuffle(dataset_records)
    n_train = int(len(dataset_records) * 0.8)
    train_data = dataset_records[:n_train]
    val_data = dataset_records[n_train:]
    
    train_loader = DataLoader(MemoryVideoDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MemoryVideoDataset(val_data), batch_size=batch_size, shuffle=False)
    
    device = torch.device('cpu')
    model = create_model(num_frames=4, image_size=64, d_model=64, num_heads=4)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.05, 1.0]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    best_state = None
    
    print("\n--- Starting Model Training (64x64 PyTorch Long-Distance Attention) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
        scheduler.step()
        train_acc = (train_correct / train_total) * 100.0 if train_total > 0 else 0.0
        train_loss = train_loss / train_total if train_total > 0 else 0.0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = (val_correct / val_total) * 100.0 if val_total > 0 else 0.0
        val_loss = val_loss / val_total if val_total > 0 else 0.0
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%")
        
        if val_acc >= best_val_acc or best_state is None:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_state,
                'val_acc': best_val_acc,
                'num_frames': 4,
                'image_size': 64,
                'd_model': 64,
                'num_heads': 4
            }, BEST_MODEL_PATH)
            
    print(f"\n[Success] Best Model checkpoint saved to: {BEST_MODEL_PATH} (Validation Accuracy: {best_val_acc:.1f}%)")
    
    # Initialize SQLite and record training history
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
        real_count=real_count,
        fake_count=fake_count,
        status="completed"
    )
    
    # Run immediate validation on key test videos
    print("\n--- Verifying Predictions on Key Evaluation Videos ---")
    model.eval()
    test_eval_videos = [
        ('Real User Video (v18.mp4)', os.path.join(UPLOADS_DIR, 'user_5_20261118_125805_786817_v18.mp4'), 'REAL'),
        ('Real User Video (WhatsApp_Video)', os.path.join(UPLOADS_DIR, 'user_5_20261118_125927_950430_WhatsApp_Video_2026-11-12_at_4.21.57_PM.mp4'), 'REAL'),
        ('Deepfake Video (0000_fake.mp4)', os.path.join(DATASET_DIR, 'train', 'fake', '0000_fake.mp4'), 'FAKE'),
        ('Deepfake Video (vs1.mp4)', os.path.join(DATASET_DIR, 'val', 'fake', 'vs1.mp4'), 'FAKE')
    ]
    
    for label_desc, vpath, expected in test_eval_videos:
        if os.path.exists(vpath):
            t, meta, _ = extract_and_preprocess_frames(vpath, num_frames=4, target_size=(128, 128))
            with torch.no_grad():
                logits = model(t)
                probs = torch.softmax(logits, dim=1)[0]
            real_pct = probs[0].item() * 100.0
            fake_pct = probs[1].item() * 100.0
            verdict = "REAL" if real_pct >= fake_pct else "FAKE"
            match_icon = "[PASS]" if verdict == expected else "[FAIL]"
            print(f"  {match_icon} {label_desc}")
            print(f"         Prediction: {verdict} (REAL: {real_pct:.1f}%, FAKE: {fake_pct:.1f}%) | Expected: {expected}")

if __name__ == "__main__":
    train_production_model(epochs=15, batch_size=4, lr=0.0003)
