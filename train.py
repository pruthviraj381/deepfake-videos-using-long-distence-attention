
import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from model import DeepfakeAttentionModel
from video_utils import read_video_frames


class VideoDataset(Dataset):

    def __init__(self, root, frames, image_size, augment=False):
        self.root = Path(root)
        self.frames = frames
        self.image_size = image_size
        self.augment = augment
        self.samples = []

        for class_name, label in [("real", 0), ("fake", 1)]:
            folder = self.root / class_name

            if not folder.exists():
                raise FileNotFoundError(
                    f"Folder not found: {folder}"
                )

            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in (
                    ".mp4", ".avi", ".mov",
                    ".mkv", ".mpeg", ".mpg"
                ):
                    self.samples.append((path, label))

        if not self.samples:
            raise ValueError(f"No videos found in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        video = read_video_frames(
            path,
            num_frames=self.frames,
            image_size=self.image_size
        )

        # Apply augmentation only to training videos.
        # The same transformation is applied to every frame.
        if self.augment:

            # Random horizontal flip.
            if random.random() < 0.5:
                video = torch.flip(video, dims=[3])

            # Small brightness and contrast variations.
            brightness = random.uniform(0.90, 1.10)
            contrast = random.uniform(0.90, 1.10)

            video_mean = video.mean(
                dim=(0, 2, 3),
                keepdim=True
            )

            video = (
                (video - video_mean) * contrast
                + video_mean
            )

            video = video * brightness
            video = torch.clamp(video, 0.0, 1.0)

        return video, torch.tensor(label, dtype=torch.long)


def calculate_metrics(model, loader, device, criterion):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    class_correct = [0, 0]
    class_total = [0, 0]

    with torch.no_grad():

        for videos, labels in loader:

            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1

                if predictions[i].item() == label:
                    class_correct[label] += 1

    average_loss = total_loss / max(total, 1)
    accuracy = 100 * correct / max(total, 1)

    real_accuracy = (
        100 * class_correct[0] / max(class_total[0], 1)
    )

    fake_accuracy = (
        100 * class_correct[1] / max(class_total[1], 1)
    )

    balanced_accuracy = (real_accuracy + fake_accuracy) / 2

    return (
        average_loss,
        accuracy,
        real_accuracy,
        fake_accuracy,
        balanced_accuracy
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)

    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)

    device = torch.device("cpu")

    print("Device:", device)
    print("Loading videos...")

    # Augmentation is enabled only for training.
    train_dataset = VideoDataset(
        Path(args.data_dir) / "train",
        args.frames,
        args.image_size,
        augment=True
    )

    # Separate unaugmented dataset for accurate training metrics.
    train_eval_dataset = VideoDataset(
        Path(args.data_dir) / "train",
        args.frames,
        args.image_size,
        augment=False
    )

    # Validation data is never augmented.
    val_dataset = VideoDataset(
        Path(args.data_dir) / "val",
        args.frames,
        args.image_size,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print("Training videos:", len(train_dataset))
    print("Validation videos:", len(val_dataset))
    print("Training augmentation: Enabled")
    print("Validation augmentation: Disabled")

    model = DeepfakeAttentionModel().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.001
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Save separately. Do not overwrite the original model.
    checkpoint_path = (
        checkpoint_dir / "groupnorm_augmentation_best_model.pth"
    )

    best_balanced_accuracy = -1.0
    best_val_loss = float("inf")

    patience = 5
    epochs_without_improvement = 0

    for epoch in range(args.epochs):

        model.train()
        running_loss = 0.0

        print(f"\nStarting Epoch {epoch + 1}/{args.epochs}")

        for videos, labels in train_loader:

            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            running_loss += loss.item()

        training_loss = (
            running_loss / max(len(train_loader), 1)
        )

        train_metrics = calculate_metrics(
            model,
            train_eval_loader,
            device,
            criterion
        )

        val_metrics = calculate_metrics(
            model,
            val_loader,
            device,
            criterion
        )

        val_loss = val_metrics[0]
        balanced_accuracy = val_metrics[4]

        print("\nTraining Results")
        print("Training Loss:", round(training_loss, 4))
        print("Training Accuracy:", round(train_metrics[1], 2), "%")

        print("\nValidation Results")
        print("Validation Loss:", round(val_loss, 4), "%")
        print("Validation Accuracy:", round(val_metrics[1], 2), "%")
        print("REAL Accuracy:", round(val_metrics[2], 2), "%")
        print("FAKE Accuracy:", round(val_metrics[3], 2), "%")
        print("Balanced Accuracy:", round(balanced_accuracy, 2), "%")

        scheduler.step(balanced_accuracy)

        current_lr = optimizer.param_groups[0]["lr"]
        print("Learning Rate:", current_lr)

        # Save the best experimental model.
        improved = (
            balanced_accuracy > best_balanced_accuracy
            or (
                balanced_accuracy == best_balanced_accuracy
                and val_loss < best_val_loss
            )
        )

        if improved:

            best_balanced_accuracy = balanced_accuracy
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "frames": args.frames,
                    "image_size": args.image_size,
                    "labels": {
                        "REAL": 0,
                        "FAKE": 1
                    },
                    "best_balanced_accuracy": best_balanced_accuracy,
                    "best_validation_loss": best_val_loss,
                    "architecture": "GroupNorm",
                    "augmentation": True
                },
                checkpoint_path
            )

            print("\nNew best augmentation model saved!")

        else:
            epochs_without_improvement += 1
            print(
                "No improvement:",
                epochs_without_improvement,
                "/",
                patience
            )

        if epochs_without_improvement >= patience:
            print("\nEarly stopping activated.")
            break

    print("\nTraining completed.")
    print(
        "Best balanced accuracy:",
        round(best_balanced_accuracy, 2),
        "%"
    )
    print("Saved model:", checkpoint_path)


if __name__ == "__main__":
    main()
