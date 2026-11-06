
from pathlib import Path
import cv2

folders = [
    "dataset/train/real",
    "dataset/train/fake",
    "dataset/val/real",
    "dataset/val/fake",
    "dataset/test/real",
    "dataset/test/fake"
]

extensions = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg")

for folder in folders:
    print("\n", "=" * 45)
    print("FOLDER:", folder)

    files = [
        p for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    print("Video count:", len(files))

    for path in sorted(files):
        cap = cv2.VideoCapture(str(path))

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        duration = frames / fps if fps > 0 else 0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()

        print(
            f"{path.name}: "
            f"{duration:.2f} seconds, "
            f"{width}x{height}, "
            f"{fps:.2f} FPS"
        )
