import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from predict import predict_video, is_model_trained

print(f"Model trained status: {is_model_trained()}")

# Test Real video
real_video = os.path.join(BASE_DIR, 'dataset', 'test', 'real', 'test_real_01.mp4')
print(f"\n--- Testing REAL Video Inference ({os.path.basename(real_video)}) ---")
res_real = predict_video(real_video, preview_save_dir=os.path.join(BASE_DIR, 'static', 'previews'))
print("Result for Real:", res_real)

# Test Fake video
fake_video = os.path.join(BASE_DIR, 'dataset', 'test', 'fake', 'test_fake_01.mp4')
print(f"\n--- Testing FAKE Video Inference ({os.path.basename(fake_video)}) ---")
res_fake = predict_video(fake_video, preview_save_dir=os.path.join(BASE_DIR, 'static', 'previews'))
print("Result for Fake:", res_fake)
