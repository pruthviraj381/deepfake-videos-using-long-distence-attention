import cv2
import numpy as np
import torch

VIDEO_EXTENSIONS = (
    ".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"
)

def read_video_frames(path, num_frames=4, image_size=64):

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        raise ValueError(f"Video has no readable frames: {path}")

    indices = np.linspace(
        0, total - 1, num_frames, dtype=int
    )

    frames = []

    for idx in indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if not ok or frame is None:
            continue

        frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        frame = cv2.resize(
            frame,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA
        )

        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"No readable frames in: {path}")

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    arr = np.stack(frames[:num_frames], axis=0)

    video_tensor = torch.from_numpy(arr).permute(
        0, 3, 1, 2
    ).contiguous()

    return video_tensor