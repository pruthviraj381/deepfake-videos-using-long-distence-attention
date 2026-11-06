
from pathlib import Path
import torch
from model import DeepfakeAttentionModel
from video_utils import read_video_frames

# Load trained model
checkpoint = torch.load(
    "checkpoints/groupnorm_best_model.pth",
    map_location="cpu",
    weights_only=False
)

model = DeepfakeAttentionModel()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

correct = 0
total = 0

# Test validation videos
for category in ["real", "fake"]:

    folder = Path("dataset/val") / category

    if not folder.exists():
        print("Folder not found:", folder)
        continue

    for video in folder.glob("*.mp4"):

        x = read_video_frames(
            video,
            checkpoint["frames"],
            checkpoint["image_size"]
        ).unsqueeze(0)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]

        predicted = "fake" if probs.argmax().item() == 1 else "real"

        actual = category

        if predicted == actual:
            correct += 1

        total += 1

        print(
            video.name,
            "| Actual:", actual.upper(),
            "| Predicted:", predicted.upper(),
            "| REAL:", round(probs[0].item() * 100, 2),
            "| FAKE:", round(probs[1].item() * 100, 2)
        )

print("\n========== VALIDATION RESULTS ==========")
print("Total videos:", total)
print("Correct predictions:", correct)
print("Incorrect predictions:", total - correct)

if total > 0:
    accuracy = (correct / total) * 100
    print("Validation Accuracy:", round(accuracy, 2), "%")
