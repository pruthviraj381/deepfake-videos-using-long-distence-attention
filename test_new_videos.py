
from pathlib import Path
import torch
import torch.nn.functional as F

from model import DeepfakeAttentionModel
from video_utils import read_video_frames

BASE_DIR = Path(__file__).parent
CHECKPOINT = BASE_DIR / "checkpoints" / "groupnorm_best_model.pth"

VIDEO_EXTENSIONS = (
    ".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"
)

device = torch.device("cpu")

# Load trained model
data = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

frames = data.get("frames", 4)
image_size = data.get("image_size", 64)

model = DeepfakeAttentionModel()
model.load_state_dict(data["model_state_dict"], strict=True)

print("GroupNorm model loaded successfully!")

model.to(device)
model.eval()

print("\nDEEPFAKE VIDEO TESTING")
print("----------------------")

correct = 0
total = 0

for category in ["real", "fake"]:

    folder = BASE_DIR / "dataset" / "test" / category

    videos = [
        p for p in folder.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS
    ]

    print(f"\nTesting {category.upper()} videos: {len(videos)}")

    for video in videos:

        try:
            video_frames = read_video_frames(
                str(video),
                num_frames=frames,
                image_size=image_size
            )

            video_frames = video_frames.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(video_frames)
                probabilities = F.softmax(output, dim=1)[0]

            predicted = (
                "FAKE" if probabilities[1] > probabilities[0]
                else "REAL"
            )

            actual = category.upper()

            if predicted == actual:
                correct += 1
                status = "CORRECT"
            else:
                status = "WRONG"

            total += 1

            print("\nVideo:", video.name)
            print("Actual:", actual)
            print("Predicted:", predicted)
            print(f"REAL: {probabilities[0].item()*100:.2f}%")
            print(f"FAKE: {probabilities[1].item()*100:.2f}%")
            print("Result:", status)

        except Exception as e:
            print("Error:", video.name, e)

print("\n----------------------")
print("TOTAL VIDEOS:", total)
print("CORRECT:", correct)
print("INCORRECT:", total - correct)

if total:
    print(f"TEST ACCURACY: {correct / total * 100:.2f}%")
else:
    print("No videos found.")
