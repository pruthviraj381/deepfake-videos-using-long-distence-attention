
import cv2
from pathlib import Path

video_path = Path("dataset/val/fake/vs17.mp4")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Cannot open video")
    raise SystemExit

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

indices = [0, total // 3, (2 * total) // 3, total - 1]

print("Video:", video_path.name)
print("Total frames:", total)

for index in indices:

    cap.set(cv2.CAP_PROP_POS_FRAMES, index)

    success, frame = cap.read()

    if not success:
        print("Frame", index, "- Cannot read")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )

    print(
        "Frame", index,
        "- Faces detected:", len(faces)
    )

cap.release()
print("Face detection check completed.")
