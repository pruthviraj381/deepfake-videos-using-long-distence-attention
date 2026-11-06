
from pathlib import Path
import torch

from model import DeepfakeAttentionModel
from video_utils import read_video_frames, VIDEO_EXTENSIONS

torch.set_num_threads(2)

device = torch.device("cpu")

checkpoint_path = (
    "checkpoints/groupnorm_augmentation_best_model.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

model = DeepfakeAttentionModel()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

frames = checkpoint["frames"]
image_size = checkpoint["image_size"]

correct = 0
total = 0

class_correct = [0, 0]
class_total = [0, 0]

for class_name, actual_label in [("real", 0), ("fake", 1)]:

    folder = Path("dataset/test") / class_name

    if not folder.exists():
        print("Test folder not found:", folder)
        continue

    print(f"\n--- {class_name.upper()} TEST VIDEOS ---")

    for path in sorted(folder.iterdir()):

        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        video = read_video_frames(
            path, frames, image_size
        ).unsqueeze(0)

        with torch.no_grad():
            output = model(video)
            probs = torch.softmax(output, dim=1)[0]

        predicted = int(output.argmax(dim=1).item())

        result = "FAKE" if predicted == 1 else "REAL"
        actual = "FAKE" if actual_label == 1 else "REAL"

        is_correct = predicted == actual_label

        correct += int(is_correct)
        total += 1

        class_total[actual_label] += 1

        if is_correct:
            class_correct[actual_label] += 1

        print(
            f"{path.name}: Actual={actual}, "
            f"Predicted={result}, "
            f"REAL={probs[0].item()*100:.1f}%, "
            f"FAKE={probs[1].item()*100:.1f}%, "
            f"Correct={is_correct}"
        )

        del video, output, probs

print("\n========== FINAL TEST RESULTS ==========")

if total > 0:

    real_accuracy = (
        100 * class_correct[0] / max(class_total[0], 1)
    )

    fake_accuracy = (
        100 * class_correct[1] / max(class_total[1], 1)
    )

    balanced_accuracy = (
        real_accuracy + fake_accuracy
    ) / 2

    print(
        f"Overall Test Accuracy: {correct}/{total} "
        f"({correct / total * 100:.2f}%)"
    )

    print("REAL Accuracy:", round(real_accuracy, 2), "%")
    print("FAKE Accuracy:", round(fake_accuracy, 2), "%")
    print("Balanced Accuracy:", round(balanced_accuracy, 2), "%")

else:
    print("No test videos were found.")
