"""
Model Training Pipeline for Deepfake Video Detection
Implements:
- PyTorch VideoDataset for train/val/test splits
- Low-memory CPU-compatible DataLoader (batch_size=2 or 4)
- Training & Validation loops with accuracy/loss computation
- Best checkpoint saving (checkpoints/best_model.pth)
- Thread-safe background training with live progress reporting for Admin UI
- SQLite training history logging
"""

import os
import glob
import time
import threading

from torch_loader import torch, nn, Dataset, DataLoader, TORCH_AVAILABLE, TORCH_ERROR
from model import create_model
from video_utils import extract_and_preprocess_frames
import database

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')

# Ensure directory structure exists
for split in ['train', 'val', 'test']:
    for category in ['real', 'fake']:
        os.makedirs(os.path.join(DATASET_DIR, split, category), exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Global thread-safe training monitor state
training_lock = threading.Lock()
training_state = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "current_step": 0,
    "total_steps": 0,
    "train_loss": 0.0,
    "train_acc": 0.0,
    "val_loss": 0.0,
    "val_acc": 0.0,
    "best_val_acc": 0.0,
    "status_message": "Ready to train",
    "error": None,
    "history": []
}


def get_dataset_counts():
    """Returns the count of real and fake videos across train, val, and test splits."""
    counts = {}
    total_videos = 0
    for split in ['train', 'val', 'test']:
        counts[split] = {}
        for cat in ['real', 'fake']:
            folder = os.path.join(DATASET_DIR, split, cat)
            exts = ['*.mp4', '*.avi', '*.mov']
            files = []
            for ext in exts:
                files.extend(glob.glob(os.path.join(folder, ext)))
                files.extend(glob.glob(os.path.join(folder, ext.upper())))
            c = len(files)
            counts[split][cat] = c
            total_videos += c
    counts['total'] = total_videos
    return counts


class DeepfakeVideoDataset(Dataset):
    """
    Custom PyTorch Dataset for loading video samples.
    Label 0: REAL
    Label 1: FAKE
    Includes in-memory frame cache to dramatically speed up multi-epoch training on CPU.
    """
    def __init__(self, split='train', num_frames=4, target_size=(64, 64)):
        self.split = split
        self.num_frames = num_frames
        self.target_size = target_size
        self.samples = []  # list of (video_path, label)
        self.cache = {}

        split_dir = os.path.join(DATASET_DIR, split)
        real_dir = os.path.join(split_dir, 'real')
        fake_dir = os.path.join(split_dir, 'fake')

        valid_exts = ('.mp4', '.avi', '.mov')

        if os.path.exists(real_dir):
            for fname in os.listdir(real_dir):
                if fname.lower().endswith(valid_exts):
                    self.samples.append((os.path.join(real_dir, fname), 0))

        if os.path.exists(fake_dir):
            for fname in os.listdir(fake_dir):
                if fname.lower().endswith(valid_exts):
                    self.samples.append((os.path.join(fake_dir, fname), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx], torch.tensor(self.samples[idx][1], dtype=torch.long)

        video_path, label = self.samples[idx]
        try:
            tensor_frames, _, _ = extract_and_preprocess_frames(
                video_path=video_path,
                num_frames=self.num_frames,
                target_size=self.target_size
            )
            # Remove batch dim: (1, T, 3, 64, 64) -> (T, 3, 64, 64)
            tensor_frames = tensor_frames.squeeze(0)
        except Exception:
            # Fallback zero tensor if video read encounters corrupted file
            tensor_frames = torch.zeros((self.num_frames, 3, self.target_size[0], self.target_size[1]), dtype=torch.float32)

        self.cache[idx] = tensor_frames
        return tensor_frames, torch.tensor(label, dtype=torch.long)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for step, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

        with training_lock:
            training_state["current_step"] = step + 1
            training_state["total_steps"] = len(dataloader)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = (correct / max(total, 1)) * 100.0
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Runs validation evaluation."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    val_loss = running_loss / max(total, 1)
    val_acc = (correct / max(total, 1)) * 100.0
    return val_loss, val_acc


def run_training_pipeline(epochs=5, batch_size=2, lr=0.0003, admin_id=1):
    """
    Executes the training process and updates the global status dictionary.
    Safe for 4GB RAM systems running on CPU.
    """
    global training_state

    with training_lock:
        training_state["is_training"] = True
        training_state["current_epoch"] = 0
        training_state["total_epochs"] = epochs
        training_state["status_message"] = "Preparing datasets..."
        training_state["error"] = None
        training_state["history"] = []

    device = torch.device('cpu')

    try:
        train_dataset = DeepfakeVideoDataset(split='train', num_frames=4, target_size=(64, 64))
        val_dataset = DeepfakeVideoDataset(split='val', num_frames=4, target_size=(64, 64))

        if len(train_dataset) == 0:
            raise ValueError(
                "No training videos found in dataset/train/real and dataset/train/fake. "
                "Please add sample videos to start training."
            )

        # In case validation set is empty, fallback to training set for validation
        if len(val_dataset) == 0:
            val_dataset = train_dataset

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0  # Safe on Windows CPU
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )

        model = create_model(num_frames=4, image_size=64, d_model=64, num_heads=4)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        best_val_acc = 0.0
        best_model_state = None

        with training_lock:
            training_state["status_message"] = f"Training started on {len(train_dataset)} samples..."

        for epoch in range(1, epochs + 1):
            with training_lock:
                training_state["current_epoch"] = epoch
                training_state["status_message"] = f"Training Epoch {epoch}/{epochs}..."

            t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            v_loss, v_acc = evaluate(model, val_loader, criterion, device)

            # Check if this is the best model so far
            if v_acc >= best_val_acc or best_model_state is None:
                best_val_acc = v_acc
                best_model_state = model.state_dict().copy()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'val_acc': best_val_acc,
                    'num_frames': 4,
                    'image_size': 64,
                    'd_model': 64,
                    'num_heads': 4
                }, BEST_MODEL_PATH)

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(t_loss, 4),
                "train_acc": round(t_acc, 2),
                "val_loss": round(v_loss, 4),
                "val_acc": round(v_acc, 2)
            }

            with training_lock:
                training_state["train_loss"] = round(t_loss, 4)
                training_state["train_acc"] = round(t_acc, 2)
                training_state["val_loss"] = round(v_loss, 4)
                training_state["val_acc"] = round(v_acc, 2)
                training_state["best_val_acc"] = round(best_val_acc, 2)
                training_state["history"].append(epoch_record)
                training_state["status_message"] = f"Epoch {epoch}/{epochs} finished (Val Acc: {round(v_acc, 1)}%)"

        # Log training completion into SQLite
        database.init_db()
        counts = get_dataset_counts()
        database.log_training_run(
            admin_id=admin_id,
            epochs=epochs,
            batch_size=batch_size,
            train_loss=training_state["train_loss"],
            train_accuracy=training_state["train_acc"],
            val_loss=training_state["val_loss"],
            val_accuracy=training_state["val_acc"],
            checkpoint_path=BEST_MODEL_PATH,
            real_count=counts['train']['real'] + counts['val']['real'],
            fake_count=counts['train']['fake'] + counts['val']['fake'],
            status="completed"
        )

        with training_lock:
            training_state["is_training"] = False
            training_state["status_message"] = f"Training completed successfully! Model checkpoint saved to {os.path.basename(BEST_MODEL_PATH)}."

    except Exception as e:
        with training_lock:
            training_state["is_training"] = False
            training_state["error"] = str(e)
            training_state["status_message"] = f"Training failed: {str(e)}"
        print(f"[Training Error]: {str(e)}")


def start_training_background(epochs=5, batch_size=2, lr=0.0003, admin_id=1):
    """Launches the training pipeline in a detached background thread."""
    if not TORCH_AVAILABLE:
        return False, (
            f"PyTorch native DLL missing ({TORCH_ERROR}). "
            "Please run 'C:\\Users\\RRC\\Downloads\\vc_redist.x64.exe' to install Microsoft Visual C++ 2015-2022 Redistributable."
        )

    with training_lock:
        if training_state["is_training"]:
            return False, "Training is already in progress."

    thread = threading.Thread(
        target=run_training_pipeline,
        kwargs={"epochs": epochs, "batch_size": batch_size, "lr": lr, "admin_id": admin_id},
        daemon=True
    )
    thread.start()
    return True, "Training started in background."


def get_training_status():
    """Returns current thread-safe training progress."""
    with training_lock:
        return dict(training_state)


if __name__ == "__main__":
    print(f"Dataset counts: {get_dataset_counts()}")
