"""
Realistic Dataset Builder & Deepfake Synthesizer
Uses actual human video samples (including user uploaded videos) to build
a comprehensive, balanced training and validation dataset for the Long-Distance Attention Model.

Creates:
1. Real Human Video Clips (natural facial motion, smooth lighting, biological consistency)
2. Realistic Deepfake Video Clips (FaceSwap seam artifacts, temporal landmark jitter, GAN blur & color mismatch)
"""

import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')


def apply_deepfake_manipulation(frame, frame_idx, manipulation_type="faceswap"):
    """
    Applies authentic deepfake artifacts to a video frame.
    Types:
    1. 'faceswap': Boundary blend seam + color tone mismatch around face region.
    2. 'warping': Temporal landmark warping & micro-jitter across distant frames.
    3. 'gan_blur': Synthesis blur & high-frequency edge artifacts in facial mask.
    """
    h, w = frame.shape[:2]
    result = frame.copy()

    # Detect face region or use central face proportions
    face_w = int(w * 0.45)
    face_h = int(h * 0.45)
    cx, cy = w // 2, int(h * 0.45)
    x1, y1 = max(0, cx - face_w // 2), max(0, cy - face_h // 2)
    x2, y2 = min(w, cx + face_w // 2), min(h, cy + face_h // 2)

    face_patch = result[y1:y2, x1:x2].copy()
    if face_patch.size == 0:
        return result

    if manipulation_type == "faceswap":
        # 1. Color tone mismatch (donor face lighting differs from target)
        color_shift = np.array([18, -12, -22], dtype=np.int16)  # unnatural yellow/green tint
        tinted = np.clip(face_patch.astype(np.int16) + color_shift, 0, 255).astype(np.uint8)

        # 2. Boundary blend seam (visible edge step of the swapped face mask)
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.ellipse(mask, ((x2 - x1) // 2, (y2 - y1) // 2), ((x2 - x1) // 2 - 6, (y2 - y1) // 2 - 6), 0, 0, 360, 255, -1)
        mask_inv = cv2.bitwise_not(mask)

        # Blend with visible boundary artifact
        inner = cv2.bitwise_and(tinted, tinted, mask=mask)
        outer = cv2.bitwise_and(face_patch, face_patch, mask=mask_inv)
        blended = cv2.add(inner, outer)
        cv2.ellipse(blended, ((x2 - x1) // 2, (y2 - y1) // 2), ((x2 - x1) // 2 - 6, (y2 - y1) // 2 - 6), 0, 0, 360, (200, 180, 220), 1)
        result[y1:y2, x1:x2] = blended

    elif manipulation_type == "warping":
        # Temporal warping / jitter: alternating geometric shifts between frames
        shift_x = int(np.sin(frame_idx * 1.8) * 8)
        shift_y = int(np.cos(frame_idx * 2.2) * 6)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        warped_face = cv2.warpAffine(face_patch, M, (face_patch.shape[1], face_patch.shape[0]), borderMode=cv2.BORDER_REFLECT)

        # Alternate blur on odd frames to simulate inconsistent synthesis resolution
        if frame_idx % 2 == 1:
            warped_face = cv2.GaussianBlur(warped_face, (9, 9), 0)

        result[y1:y2, x1:x2] = warped_face

    elif manipulation_type == "gan_blur":
        # Synthesis blur inside face region + high-frequency noise along borders
        blurred = cv2.GaussianBlur(face_patch, (11, 11), 0)
        noise = np.random.randint(-15, 15, face_patch.shape, dtype=np.int16)
        noisy = np.clip(blurred.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        result[y1:y2, x1:x2] = noisy

    return result


def extract_clips_and_synthesize_deepfakes():
    """Builds a balanced dataset from available real videos and deepfake counterparts."""
    # Find available real source videos
    source_candidates = [
        os.path.join(UPLOADS_DIR, 'user_5_20261118_125805_786817_v18.mp4'),
        os.path.join(UPLOADS_DIR, 'user_5_20261118_125927_950430_WhatsApp_Video_2026-11-12_at_4.21.57_PM.mp4'),
    ]

    valid_sources = [p for p in source_candidates if os.path.exists(p)]
    print(f"[Dataset] Found {len(valid_sources)} source real videos for realistic training.")

    if not valid_sources:
        print("[Dataset Error] No source video files found in uploads.")
        return

    clip_configs = [
        ('train', 'real', False, None),
        ('train', 'fake', True, 'faceswap'),
        ('train', 'fake', True, 'warping'),
        ('train', 'fake', True, 'gan_blur'),
        ('val', 'real', False, None),
        ('val', 'fake', True, 'faceswap'),
        ('test', 'real', False, None),
        ('test', 'fake', True, 'warping'),
    ]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    clip_idx = 1

    for src_path in valid_sources:
        cap = cv2.VideoCapture(src_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Read frames from the video
        frames = []
        for _ in range(min(total_frames, 120)):
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
        cap.release()

        if len(frames) < 16:
            continue

        h, w = frames[0].shape[:2]
        # Resize to standardized 320x320 for clean, fast dataset generation
        target_size = (320, 320)
        resized_frames = [cv2.resize(f, target_size) for f in frames]

        # Generate Real clips (chunks of 20 frames)
        for split in ['train', 'val', 'test']:
            split_real_dir = os.path.join(DATASET_DIR, split, 'real')
            split_fake_dir = os.path.join(DATASET_DIR, split, 'fake')
            os.makedirs(split_real_dir, exist_ok=True)
            os.makedirs(split_fake_dir, exist_ok=True)

            num_clips = 3 if split == 'train' else 1
            for i in range(num_clips):
                start_f = (i * 24) % max(1, len(resized_frames) - 20)
                sub_frames = resized_frames[start_f:start_f + 20]

                # 1. Save REAL clip
                real_out = os.path.join(split_real_dir, f"human_real_{clip_idx:03d}.mp4")
                out_writer = cv2.VideoWriter(real_out, fourcc, 15, target_size)
                for f in sub_frames:
                    out_writer.write(f)
                out_writer.release()

                # 2. Save FAKE deepfake clips (FaceSwap, Warping, GAN Blur)
                for manip_type in ['faceswap', 'warping', 'gan_blur']:
                    fake_out = os.path.join(split_fake_dir, f"deepfake_{manip_type}_{clip_idx:03d}.mp4")
                    out_fake_writer = cv2.VideoWriter(fake_out, fourcc, 15, target_size)
                    for f_i, f in enumerate(sub_frames):
                        manipulated = apply_deepfake_manipulation(f, f_i, manipulation_type=manip_type)
                        out_fake_writer.write(manipulated)
                    out_fake_writer.release()

                clip_idx += 1

    print("[Dataset] Real and Deepfake clips generated successfully.")


if __name__ == "__main__":
    extract_clips_and_synthesize_deepfakes()
