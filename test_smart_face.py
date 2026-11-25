import cv2
import numpy as np
import torch
from predict import load_model

def get_smart_4_face_frames(video_path, target_size=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    quarters = [
        (0, total_frames // 4),
        (total_frames // 4, 2 * total_frames // 4),
        (2 * total_frames // 4, 3 * total_frames // 4),
        (3 * total_frames // 4, total_frames)
    ]
    
    processed = []
    faces_found = 0
    
    for q_start, q_end in quarters:
        mid = (q_start + q_end) // 2
        # Search candidate indices around midpoint in this quarter
        offsets = [0, 5, -5, 10, -10, 15, -15, 20, -20, 25, -25, 30, -30, 40, -40, 50, -50]
        best_face_img = None
        
        for off in offsets:
            idx = mid + off
            if q_start <= idx < q_end:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(24, 24))
                    if len(faces) > 0:
                        largest = max(faces, key=lambda b: b[2] * b[3])
                        x, y, w, h = largest
                        # crop face with margin
                        m_x, m_y = int(w * 0.15), int(h * 0.15)
                        x1 = max(0, x - m_x)
                        y1 = max(0, y - m_y)
                        x2 = min(frame.shape[1], x + w + m_x)
                        y2 = min(frame.shape[0], y + h + m_y)
                        best_face_img = frame[y1:y2, x1:x2]
                        faces_found += 1
                        break
        
        if best_face_img is None:
            # Fallback to mid frame center crop
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
            h, w = frame.shape[:2]
            md = min(h, w)
            best_face_img = frame[(h-md)//2:(h-md)//2+md, (w-md)//2:(w-md)//2+md]
            
        resized = cv2.resize(best_face_img, target_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm = (rgb.astype(np.float32) / 255.0 - mean) / std
        processed.append(np.transpose(norm, (2, 0, 1)))
        
    cap.release()
    tensor = torch.from_numpy(np.stack(processed, axis=0)).unsqueeze(0).float()
    return tensor, faces_found

v1 = r'c:\Users\RRC\Desktop\project\deepfake_detection\uploads\user_5_20261118_125927_950430_WhatsApp_Video_2026-11-12_at_4.21.57_PM.mp4'
v2 = r'c:\Users\RRC\Desktop\project\deepfake_detection\uploads\user_5_20261118_125805_786817_v18.mp4'

model = load_model()

for vp, name in [(v1, 'WhatsApp_Video'), (v2, 'v18')]:
    tensor, faces = get_smart_4_face_frames(vp)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    print(f"\nVideo: {name}")
    print(f"Faces found: {faces}/4")
    print(f"Logits: {logits[0].tolist()}")
    print(f"Probabilities: REAL={probs[0].item()*100:.2f}%, FAKE={probs[1].item()*100:.2f}%")
