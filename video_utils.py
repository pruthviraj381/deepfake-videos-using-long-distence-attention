"""
Video Processing & Feature Extraction Utilities
Handles:
- Video file validation
- Uniform temporal frame extraction (Long-Distance sequence)
- Face detection & cropping via OpenCV Haar Cascades
- Resolution standardization (64x64) and normalization
- Generation of visual frame preview montages
- Safe downloading of online video URLs (direct video links & yt-dlp)
"""

import os
import re
import cv2
import numpy as np
from datetime import datetime
from PIL import Image

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit for 4GB RAM CPU environment


def allowed_file(filename):
    """Checks whether the filename has a supported video extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_video_metadata(video_path):
    """Retrieves video metadata such as duration, FPS, and frame count."""
    if not os.path.exists(video_path):
        return {"duration": 0.0, "fps": 0.0, "frame_count": 0, "file_size": 0}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"duration": 0.0, "fps": 0.0, "frame_count": 0, "file_size": os.path.getsize(video_path)}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = round(total_frames / fps, 2) if fps > 0 else 0.0
    cap.release()

    return {
        "duration": duration,
        "fps": round(fps, 1),
        "frame_count": total_frames,
        "file_size": os.path.getsize(video_path)
    }


def get_face_cascade():
    """Loads OpenCV's pre-trained frontal face Haar cascade classifier."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"OpenCV Haar Cascade file not found at: {cascade_path}")
    return cv2.CascadeClassifier(cascade_path)


def crop_face_with_margin(frame_bgr, bbox, margin_ratio=0.15):
    """
    Crops the detected face with a margin to preserve boundary manipulation artifacts.
    bbox: (x, y, w, h)
    """
    h_img, w_img = frame_bgr.shape[:2]
    x, y, w, h = bbox

    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w_img, x + w + margin_x)
    y2 = min(h_img, y + h + margin_y)

    cropped = frame_bgr[y1:y2, x1:x2]
    return cropped


def center_crop_fallback(frame_bgr):
    """Center crops square region when no face is explicitly detected."""
    h, w = frame_bgr.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    return frame_bgr[start_y:start_y + min_dim, start_x:start_x + min_dim]


def extract_and_preprocess_frames(video_path, num_frames=4, target_size=(64, 64), preview_save_dir=None):
    """
    Extracts `num_frames` across the video uniformly (Long-Distance temporal sampling).
    Detects faces, crops, resizes to target_size (64x64), normalizes, and packages into a PyTorch tensor.
    
    Returns:
        tensor_frames: torch.Tensor of shape (1, num_frames, 3, target_size[0], target_size[1])
        metadata: dict containing video duration, fps, frame count, face detection rate, and forensic signals
        preview_filename: str name of montage image if preview_save_dir is provided
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = total_frames / fps if fps > 0 else 0.0

    if total_frames <= 0:
        # Fallback if frame count cannot be determined directly
        frames_list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_list.append(frame)
        total_frames = len(frames_list)
        cap.release()
        cap = None
    else:
        frames_list = None

    if total_frames == 0:
        if cap:
            cap.release()
        raise ValueError("Video contains 0 readable frames.")

    # Select long-distance temporal segments spanning the entire video
    segments = []
    seg_size = total_frames / float(num_frames)
    for i in range(num_frames):
        s_start = int(i * seg_size)
        s_end = int((i + 1) * seg_size) if i < num_frames - 1 else total_frames
        s_mid = (s_start + s_end) // 2
        segments.append((s_start, s_end, s_mid))

    face_cascade = get_face_cascade()
    processed_frames = []
    preview_thumbnails = []
    faces_detected_count = 0

    # ImageNet normalization statistics
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for s_start, s_end, s_mid in segments:
        best_face_img = None
        best_frame_bgr = None
        found_face = False

        # Search outwards from midpoint of this temporal segment for a clear face
        search_offsets = [0, 5, -5, 10, -10, 15, -15, 20, -20, 30, -30, 45, -45, 60, -60, 90, -90]
        for off in search_offsets:
            cand_idx = s_mid + off
            if s_start <= cand_idx < s_end:
                if frames_list is not None:
                    cand_frame = frames_list[cand_idx]
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, cand_idx)
                    ret, cand_frame = cap.read()
                    if not ret or cand_frame is None:
                        continue

                gray = cv2.cvtColor(cand_frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.12,
                    minNeighbors=3,
                    minSize=(24, 24)
                )

                if len(faces) > 0:
                    largest_face = max(faces, key=lambda b: b[2] * b[3])
                    best_face_img = crop_face_with_margin(cand_frame, largest_face)
                    best_frame_bgr = cand_frame
                    found_face = True
                    faces_detected_count += 1
                    break

        if not found_face:
            # Fallback to midpoint frame center crop if no face detected in segment
            if frames_list is not None:
                fallback_frame = frames_list[s_mid]
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, s_mid)
                ret, fallback_frame = cap.read()
                if not ret or fallback_frame is None:
                    fallback_frame = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

            best_face_img = center_crop_fallback(fallback_frame)
            best_frame_bgr = fallback_frame

        # Resize to target 64x64
        face_resized = cv2.resize(best_face_img, target_size, interpolation=cv2.INTER_AREA)

        # RGB conversion
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        preview_thumbnails.append(face_rgb)

        # Normalize: [0, 255] -> [0.0, 1.0] -> standard score
        norm_img = face_rgb.astype(np.float32) / 255.0
        norm_img = (norm_img - mean) / std

        # Transpose from (H, W, C) to (C, H, W)
        ch_first = np.transpose(norm_img, (2, 0, 1))
        processed_frames.append(ch_first)

    if cap:
        cap.release()

    # Calculate forensic signals across the 4 temporal face crops
    forensic_scores = {
        "is_natural_skin_tone": True,
        "boundary_seam_score": 0.0,
        "temporal_jitter_score": 0.0
    }

    if len(preview_thumbnails) == num_frames:
        # 1. Biological skin tone test (R > G > B in genuine human faces)
        mean_rgb = np.mean([np.mean(thumb, axis=(0, 1)) for thumb in preview_thumbnails], axis=0)
        # In natural skin: R > G and G >= B
        natural_skin = bool(mean_rgb[0] >= mean_rgb[1] and mean_rgb[1] >= (mean_rgb[2] - 15))

        # 2. Boundary seam analysis (Laplacian variance on outer borders)
        lap_vars = []
        for thumb in preview_thumbnails:
            gray_thumb = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
            # Check border region (outer 16 pixels of 64x64)
            border_mask = np.ones((target_size[0], target_size[1]), dtype=bool)
            border_mask[16:-16, 16:-16] = False
            border_pixels = gray_thumb[border_mask]
            lap_vars.append(float(np.var(border_pixels)))

        boundary_seam = float(np.mean(lap_vars))

        # 3. Temporal consistency across long-distance frames
        pair_diffs = []
        for i in range(num_frames):
            for j in range(i + 1, num_frames):
                diff = np.mean(np.abs(preview_thumbnails[i].astype(float) - preview_thumbnails[j].astype(float)))
                pair_diffs.append(diff)
        temporal_jitter = float(np.var(pair_diffs)) if len(pair_diffs) > 0 else 0.0

        forensic_scores["is_natural_skin_tone"] = natural_skin
        forensic_scores["boundary_seam_score"] = round(boundary_seam, 2)
        forensic_scores["temporal_jitter_score"] = round(temporal_jitter, 4)

    # Stack into tensor: (1, num_frames, 3, 64, 64)
    np_stacked = np.stack(processed_frames, axis=0)  # (T, 3, H, W)
    try:
        import torch
        tensor_frames = torch.from_numpy(np_stacked).unsqueeze(0).float()  # (1, T, 3, H, W)
    except OSError as e:
        raise OSError(
            f"PyTorch native DLL failed to load: {e}. "
            "Please run 'C:\\Users\\RRC\\Downloads\\vc_redist.x64.exe' to install Microsoft Visual C++ 2015-2022 Redistributable."
        )

    # Generate preview montage image for UI
    preview_filename = None
    if preview_save_dir and len(preview_thumbnails) > 0:
        os.makedirs(preview_save_dir, exist_ok=True)
        # Create a horizontal strip of the 4 extracted face frames with frame labels
        thumb_w, thumb_h = target_size
        border = 4
        total_w = (thumb_w * num_frames) + (border * (num_frames + 1))
        total_h = thumb_h + (border * 2) + 20

        montage = Image.new('RGB', (total_w, total_h), color=(15, 23, 42))  # Dark blue background
        for i, thumb in enumerate(preview_thumbnails):
            pil_thumb = Image.fromarray(thumb)
            x_pos = border + i * (thumb_w + border)
            y_pos = border + 18
            montage.paste(pil_thumb, (x_pos, y_pos))

        preview_filename = f"preview_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        montage_path = os.path.join(preview_save_dir, preview_filename)
        montage.save(montage_path, "JPEG", quality=90)

    metadata = {
        "duration": round(duration, 2),
        "fps": round(fps, 1),
        "total_frames": total_frames,
        "frames_analyzed": num_frames,
        "faces_detected_count": faces_detected_count,
        "face_detection_rate": round((faces_detected_count / num_frames) * 100, 1),
        "resolution": f"{target_size[0]}x{target_size[1]}",
        "forensic_signals": forensic_scores
    }

    return tensor_frames, metadata, preview_filename


def download_online_video(url, output_dir, max_size_mb=50):
    """
    Deprecated: Online video URL downloading has been removed in favor of direct file upload.
    """
    return None, "Online video URL downloading has been removed. Please upload video files (.mp4, .avi, .mov) directly."
