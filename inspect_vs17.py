
import cv2
from pathlib import Path

video_path = Path("dataset/val/fake/vs17.mp4")

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Cannot open video")
    raise SystemExit

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

indices = [0, total // 3, (2 * total) // 3, total - 1]

for i, index in enumerate(indices):

    cap.set(cv2.CAP_PROP_POS_FRAMES, index)

    success, frame = cap.read()

    if success:
        frame = cv2.resize(frame, (64, 64))

        filename = f"vs17_frame_{i+1}.jpg"

        cv2.imwrite(filename, frame)

        print("Saved:", filename)

cap.release()

print("Frame inspection completed.")
