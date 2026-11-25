import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from predict import predict_video

fake_videos = [
    os.path.join(BASE_DIR, 'dataset', 'test', 'fake', 'test_fake_01.mp4'),
    os.path.join(BASE_DIR, 'dataset', 'val', 'fake', 'deepfake_faceswap_001.mp4'),
]

for fv in fake_videos:
    if os.path.exists(fv):
        res = predict_video(fv)
        print(f"File: {os.path.basename(fv)} -> Label: {res['prediction_label']}, Real%: {res['real_confidence']}%, Fake%: {res['fake_confidence']}%")
