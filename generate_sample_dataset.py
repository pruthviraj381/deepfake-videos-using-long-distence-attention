"""
Dataset Sample Generator for MCA Deepfake Detection Project
Generates realistic benchmark video clips (MP4) to enable immediate
training and evaluation out-of-the-box on 4GB RAM systems.

Generates:
- REAL videos: Natural facial geometry with smooth motion and consistent temporal lighting.
- FAKE videos: Deepfake face-swap simulation with boundary blending artifacts,
  color tone seam, and unnatural temporal jitter across distant frames.
"""

import os
import math
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')


def draw_face(img, center_x, center_y, eye_open=True, is_fake=False, frame_idx=0):
    """Draws synthetic facial structure with realistic proportions for OpenCV face cascade."""
    # Head contour (Oval)
    head_color = (200, 215, 235)  # Natural skin tone in BGR
    cv2.ellipse(img, (center_x, center_y), (48, 62), 0, 0, 360, head_color, -1)
    cv2.ellipse(img, (center_x, center_y), (48, 62), 0, 0, 360, (140, 155, 175), 2)

    # Hair
    hair_color = (35, 45, 55)
    cv2.ellipse(img, (center_x, center_y - 28), (48, 38), 0, 180, 360, hair_color, -1)

    # Eyes
    left_eye_pos = (center_x - 18, center_y - 12)
    right_eye_pos = (center_x + 18, center_y - 12)

    if eye_open:
        # Eye sclera
        cv2.circle(img, left_eye_pos, 7, (245, 245, 245), -1)
        cv2.circle(img, right_eye_pos, 7, (245, 245, 245), -1)
        # Pupil / Iris
        cv2.circle(img, left_eye_pos, 4, (60, 40, 20), -1)
        cv2.circle(img, right_eye_pos, 4, (60, 40, 20), -1)
    else:
        # Blinking eye slit
        cv2.line(img, (left_eye_pos[0] - 6, left_eye_pos[1]), (left_eye_pos[0] + 6, left_eye_pos[1]), (50, 40, 30), 2)
        cv2.line(img, (right_eye_pos[0] - 6, right_eye_pos[1]), (right_eye_pos[0] + 6, right_eye_pos[1]), (50, 40, 30), 2)

    # Eyebrows
    cv2.line(img, (left_eye_pos[0] - 9, left_eye_pos[1] - 10), (left_eye_pos[0] + 9, left_eye_pos[1] - 9), hair_color, 2)
    cv2.line(img, (right_eye_pos[0] - 9, right_eye_pos[1] - 9), (right_eye_pos[0] + 9, right_eye_pos[1] - 10), hair_color, 2)

    # Nose
    cv2.line(img, (center_x, center_y - 4), (center_x, center_y + 12), (160, 175, 195), 2)
    cv2.circle(img, (center_x - 4, center_y + 12), 2, (150, 165, 185), -1)
    cv2.circle(img, (center_x + 4, center_y + 12), 2, (150, 165, 185), -1)

    # Mouth / Lips
    cv2.ellipse(img, (center_x, center_y + 28), (14, 6), 0, 0, 360, (120, 130, 200), -1)

    # If FAKE deepfake video: inject simulated deepfake artifacts
    if is_fake:
        # 1. Unnatural boundary seam / color mismatch mask around swapped face region
        seam_color = (180, 230, 210) if (frame_idx % 2 == 0) else (220, 190, 240)
        cv2.rectangle(img, (center_x - 30, center_y - 25), (center_x + 30, center_y + 36), seam_color, 1)

        # 2. Inconsistent blurring / warping artifacts across alternating frames
        if frame_idx % 2 == 1:
            sub_region = img[max(0, center_y - 20):min(img.shape[0], center_y + 35),
                             max(0, center_x - 25):min(img.shape[1], center_x + 25)]
            if sub_region.size > 0:
                blurred = cv2.GaussianBlur(sub_region, (7, 7), 0)
                img[max(0, center_y - 20):min(img.shape[0], center_y + 35),
                    max(0, center_x - 25):min(img.shape[1], center_x + 25)] = blurred

        # 3. High-frequency digital noise / pixelation around facial borders
        noise = np.random.randint(-18, 18, (30, 60, 3), dtype=np.int16)
        patch = img[center_x - 15:center_x + 15, center_y - 30:center_y + 30].astype(np.int16)
        if patch.shape == noise.shape:
            patch = np.clip(patch + noise, 0, 255).astype(np.uint8)
            img[center_x - 15:center_x + 15, center_y - 30:center_y + 30] = patch


def create_sample_video(output_path, is_fake=False, duration_sec=2.0, fps=12, width=160, height=160):
    """Writes an MP4 video clip with simulated facial movements and deepfake traits."""
    total_frames = int(duration_sec * fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    base_cx = width // 2
    base_cy = height // 2

    for f in range(total_frames):
        # Background gradient
        bg = np.full((height, width, 3), (230, 235, 240), dtype=np.uint8)

        # Natural gentle head sway
        t = f / float(total_frames)
        offset_x = int(math.sin(t * 2 * math.pi) * 6)
        offset_y = int(math.cos(t * 2 * math.pi) * 3)

        # Periodic blinking
        eye_open = not (f in [8, 9, 18, 19])

        draw_face(
            bg,
            center_x=base_cx + offset_x,
            center_y=base_cy + offset_y,
            eye_open=eye_open,
            is_fake=is_fake,
            frame_idx=f
        )

        out.write(bg)

    out.release()


def generate_benchmark_dataset(num_train=6, num_val=2, num_test=2):
    """Generates standard train, val, and test video clips."""
    configs = [
        ('train', num_train),
        ('val', num_val),
        ('test', num_test)
    ]

    print("[Dataset Generator] Generating sample benchmark videos...")

    for split, count in configs:
        real_folder = os.path.join(DATASET_DIR, split, 'real')
        fake_folder = os.path.join(DATASET_DIR, split, 'fake')
        os.makedirs(real_folder, exist_ok=True)
        os.makedirs(fake_folder, exist_ok=True)

        for i in range(1, count + 1):
            real_file = os.path.join(real_folder, f"{split}_real_{i:02d}.mp4")
            fake_file = os.path.join(fake_folder, f"{split}_fake_{i:02d}.mp4")

            if not os.path.exists(real_file):
                create_sample_video(real_file, is_fake=False, duration_sec=2.0)
            if not os.path.exists(fake_file):
                create_sample_video(fake_file, is_fake=True, duration_sec=2.0)

    print("[Dataset Generator] All sample video sets generated successfully!")


if __name__ == "__main__":
    generate_benchmark_dataset()
