
import torch
from pathlib import Path

files = [
    "best_model.pth",
    "best_model_backup.pth",
    "groupnorm_best_model.pth",
    "old_best_model.pth"
]

for name in files:
    path = Path("checkpoints") / name
    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    print("\nMODEL:", name)
    print("Balanced Accuracy:",
          data.get("best_balanced_accuracy", "Not available"))
    print("Validation Loss:",
          data.get("best_validation_loss", "Not available"))
    print("Labels:", data.get("labels", "Not available"))
