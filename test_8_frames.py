from pathlib import Path
import torch

from model import DeepfakeAttentionModel
from video_utils import read_video_frames

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

image_size = checkpoint["image_size"]

video_path = Path("dataset/val/fake/vs17.mp4")

for num_frames in [4, 8]:

    video = read_video_frames(
        video_path,
        num_frames,
        image_size
    ).unsqueeze(0)

    with torch.no_grad():
        output = model(video)
        probs = torch.softmax(output, dim=1)[0]

    print(f"\nFrames tested: {num_frames}")
    print(f"REAL: {probs[0].item() * 100:.2f}%")
    print(f"FAKE: {probs[1].item() * 100:.2f}%")

    del video, output, probs