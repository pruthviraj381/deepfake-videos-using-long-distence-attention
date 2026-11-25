import cv2

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
for p in ['deepfake_detection/dataset/train/real/0000.mp4', 'deepfake_detection/dataset/train/fake/0000_fake.mp4']:
    cap = cv2.VideoCapture(p)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    faces = cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) if ret else []
    print(p, "FPS:", fps, "Total frames:", total, "Faces detected:", len(faces))
    cap.release()
