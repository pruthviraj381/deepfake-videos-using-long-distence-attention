import os
import cv2

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for split in ['val', 'test']:
    for cat in ['real', 'fake']:
        d = os.path.join('deepfake_detection', 'dataset', split, cat)
        files = [f for f in os.listdir(d) if f.endswith('.mp4')][:5]
        print(f"\n--- {split}/{cat} ({len(os.listdir(d))} files) ---")
        for f in files:
            cap = cv2.VideoCapture(os.path.join(d, f))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            face_cnt = 0
            for i in range(min(total, 10)):
                ret, frame = cap.read()
                if ret:
                    faces = cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=3)
                    if len(faces) > 0:
                        face_cnt += 1
            cap.release()
            print(f"  {f}: frames={total}, face_detected_in={face_cnt}/{min(total, 10)}")
