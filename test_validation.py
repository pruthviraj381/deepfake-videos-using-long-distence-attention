
from pathlib import Path
import torch

from model import DeepfakeAttentionModel
from video_utils import read_video_frames, VIDEO_EXTENSIONS

torch.set_num_threads(2)

device = torch.device("cpu")

checkpoint = torch.load(
    "checkpoints/groupnorm_best_model.pth",
    map_location=device,
    weights_only=False
)

model = DeepfakeAttentionModel()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

frames = checkpoint["frames"]
image_size = checkpoint["image_size"]

for class_name, actual_label in [("real", 0), ("fake", 1)]:

    folder = Path("dataset/val") / class_name

    print(f"\n--- {class_name.upper()} VIDEOS ---")

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

        print(
            f"{path.name}: {result} | "
            f"REAL: {probs[0].item()*100:.1f}% | "
            f"FAKE: {probs[1].item()*100:.1f}%"
        )

        del video, output, probs
