
from pathlib import Path
import torch

from model import DeepfakeAttentionModel
from video_utils import read_video_frames, VIDEO_EXTENSIONS

CHECKPOINT = Path("checkpoints/groupnorm_best_model.pth")
VALIDATION_FOLDER = Path("dataset/val")

torch.set_num_threads(2)

data = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

print("Checkpoint:", CHECKPOINT)
print("Labels:", data["labels"])

model = DeepfakeAttentionModel()

model.load_state_dict(
    data["model_state_dict"],
    strict=True
)

model.eval()

correct = 0
total = 0

class_correct = {"REAL": 0, "FAKE": 0}
class_total = {"REAL": 0, "FAKE": 0}

for category in ["real", "fake"]:

    folder = VALIDATION_FOLDER / category

    videos = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]

    print(f"\nChecking {category.upper()} videos...")

    for video in videos:

        try:
            x = read_video_frames(
                video,
                data["frames"],
                data["image_size"]
            ).unsqueeze(0)

            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0]

            predicted = (
                "FAKE" if probs.argmax().item() == 1 else "REAL"
            )

            actual = category.upper()

            real_score = probs[0].item() * 100
            fake_score = probs[1].item() * 100

            is_correct = predicted == actual

            total += 1
            class_total[actual] += 1

            if is_correct:
                correct += 1
                class_correct[actual] += 1

            print(
                f"{video.name} | Actual: {actual} | "
                f"Predicted: {predicted} | "
                f"REAL: {real_score:.2f}% | "
                f"FAKE: {fake_score:.2f}% | "
                f"{'CORRECT' if is_correct else 'WRONG'}"
            )

        except Exception as e:
            print("ERROR:", video.name, e)

print("\n========== FINAL RESULTS ==========")

for category in ["REAL", "FAKE"]:
    count = class_total[category]

    accuracy = (
        class_correct[category] / count * 100
        if count else 0
    )

    print(
        f"{category} Accuracy: "
        f"{accuracy:.2f}% "
        f"({class_correct[category]}/{count})"
    )

if total:
    overall = correct / total * 100

    real_acc = class_correct["REAL"] / max(class_total["REAL"], 1)
    fake_acc = class_correct["FAKE"] / max(class_total["FAKE"], 1)

    print(f"\nOverall Accuracy: {overall:.2f}%")
    print(f"Total Videos: {total}")
    print(
        "Balanced Accuracy:",
        f"{(real_acc + fake_acc) * 50:.2f}%"
    )
