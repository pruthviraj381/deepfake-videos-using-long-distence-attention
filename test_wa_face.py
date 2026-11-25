import cv2
import numpy as np

vpath = 'deepfake_detection/uploads/user_5_20261118_125927_950430_WhatsApp_Video_2026-11-12_at_4.21.57_PM.mp4'
cap = cv2.VideoCapture(vpath)
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames:", total)

found = []
for i in range(0, total, 10):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if ret:
        faces = cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=3)
        if len(faces) > 0:
            found.append(i)

print(f"Frames with faces ({len(found)}/{total//10}):", found)
