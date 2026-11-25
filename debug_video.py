import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import cv2
from predict import predict_video
import database

v1 = r'c:\Users\RRC\Desktop\project\deepfake_detection\uploads\user_5_20261118_125927_950430_WhatsApp_Video_2026-11-12_at_4.21.57_PM.mp4'
v2 = r'c:\Users\RRC\Desktop\project\deepfake_detection\uploads\user_5_20261118_125805_786817_v18.mp4'

for path, name in [(v1, 'WhatsApp_Video'), (v2, 'v18')]:
    print(f"\n==================== Analyzing: {name} ====================")
    if not os.path.exists(path):
        print("File does not exist:", path)
        continue
    
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"Video Info: {w}x{h}, {fps} FPS, {total_frames} Total Frames")

    res = predict_video(path)
    print("Prediction result:", res)
