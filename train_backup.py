import argparse, random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import DeepfakeAttentionModel
from video_utils import read_video_frames, VIDEO_EXTENSIONS

LABELS = {"real": 0, "fake": 1}

class VideoDataset(Dataset):
    def __init__(self, root, frames, size):
        self.items = []
        self.frames, self.size = frames, size
        root = Path(root)
        for name, label in LABELS.items():
            folder = root / name
            if folder.exists():
                for p in folder.rglob("*"):
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                        self.items.append((p, label))
        if not self.items:
            raise RuntimeError(f"No videos found under {root}. Expected real/ and fake/ folders.")
    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        path, label = self.items[index]
        try:
            x = read_video_frames(path, self.frames, self.size)
        except Exception as e:
            raise RuntimeError(f"Video error: {path}: {e}") from e
        return x, torch.tensor(label, dtype=torch.long)

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)

def main():
    ap = argparse.ArgumentParser(description="Train compact CNN + temporal attention deepfake detector")
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=0.001)
    args = ap.parse_args()
    random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(2)
    root = Path(args.data_dir)
    train_ds = VideoDataset(root / "train", args.frames, args.image_size)
    val_root = root / "val"
    val_ds = VideoDataset(val_root, args.frames, args.image_size) if val_root.exists() and any(val_root.rglob("*")) else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0) if val_ds else None
    device = torch.device("cpu")
    model = DeepfakeAttentionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    out = Path("checkpoints"); out.mkdir(exist_ok=True)
    best = -1.0
    print(f"Training videos: {len(train_ds)} | Validation videos: {len(val_ds) if val_ds else 0}")
    print("Device: CPU. Training may take time on a 4 GB RAM laptop.")
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            running += loss.item()
            print(f"\rEpoch {epoch}/{args.epochs} - batch {step}/{len(train_loader)} - loss {loss.item():.4f}", end="")
        print(f"\nEpoch {epoch} training loss: {running / max(len(train_loader),1):.4f}")
        if val_loader:
            vl, acc = evaluate(model, val_loader, device)
            print(f"Validation loss: {vl:.4f} | Validation accuracy: {acc*100:.2f}%")
            score = acc
        else:
            print("No validation videos found; saving latest model. Add separate videos to dataset/val/ for evaluation.")
            score = 0.0
        if score >= best:
            best = score
            torch.save({"model_state_dict": model.state_dict(),
                        "frames": args.frames, "image_size": args.image_size,
                        "labels": LABELS}, out / "best_model.pth")
            print("Saved checkpoints/best_model.pth")
    print("Training complete.")

if __name__ == "__main__":
    main()
