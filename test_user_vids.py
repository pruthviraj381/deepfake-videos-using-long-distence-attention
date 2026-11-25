import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import cv2
import torch
from predict import load_model
from video_utils import extract_and_preprocess_frames

v1 = r'c:\Users\RRC\Desktop\project\deepfake_detection\uploads\user_5_20261118_125927_950430_WhatsApp_Video_2026-11-12_at_4.21.57_PM.mp4'
v2 = r'c:\Users\RRC\Desktop\project\deepfake_detection\uploads\user_5_20261118_125805_786817_v18.mp4'

model = load_model()

for vp, name in [(v1, 'WhatsApp_Video'), (v2, 'v18')]:
    print(f"\n=== Video: {name} ===")
    frames, meta, preview = extract_and_preprocess_frames(vp, num_frames=4, target_size=(128, 128))
    print("Metadata:", meta)
    with torch.no_grad():
        logits = model(frames)
        probs = torch.softmax(logits, dim=1)[0]
    print("Logits [REAL, FAKE]:", logits[0].tolist())
    print(f"Probabilities: REAL={probs[0].item()*100:.2f}%, FAKE={probs[1].item()*100:.2f}%")
