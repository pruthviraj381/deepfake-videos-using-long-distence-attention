import cv2
import numpy as np
from video_utils import extract_and_preprocess_frames

for vp, name in [
    ('deepfake_detection/uploads/user_5_20261118_125927_950430_WhatsApp_Video_2026-11-12_at_4.21.57_PM.mp4', 'WhatsApp_Video'),
    ('deepfake_detection/uploads/user_5_20261118_125805_786817_v18.mp4', 'v18')
]:
    t, meta, _ = extract_and_preprocess_frames(vp, num_frames=4, target_size=(128, 128))
    print(f"\n--- {name} ---")
    print("Metadata:", meta)
