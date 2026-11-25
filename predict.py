"""
Inference & Prediction Pipeline for Deepfake Video Detection
Uses the trained Long-Distance Attention PyTorch model.
Performs:
1. Frame extraction & Face detection (OpenCV Haar Cascade)
2. Normalization to (1, 4, 3, 64, 64)
3. CNN Feature extraction + Temporal Positional Encoding
4. Multi-Head Long-Distance Attention
5. Softmax Classification (REAL vs FAKE)
"""

import os
import time

from torch_loader import torch, F, TORCH_AVAILABLE, TORCH_ERROR
from model import create_model
from video_utils import extract_and_preprocess_frames

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
DEFAULT_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')


def is_model_trained(checkpoint_path=DEFAULT_CHECKPOINT_PATH):
    """Checks whether the trained PyTorch checkpoint file exists."""
    return os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0


def load_model(checkpoint_path=DEFAULT_CHECKPOINT_PATH, device='cpu'):
    """
    Loads model weights from checkpoint into the LongDistanceAttention architecture.
    Runs on CPU with low memory overhead.
    """
    if torch is None:
        raise OSError(
            f"PyTorch failed to load ({TORCH_ERROR}). "
            "Please install the Microsoft Visual C++ 2015-2022 Redistributable by running 'C:\\Users\\RRC\\Downloads\\vc_redist.x64.exe'."
        )

    if not is_model_trained(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at '{checkpoint_path}'. "
            "Please train the model from the Admin Panel or run train.py first."
        )

    # Initialize model with standard MCA project configuration (64x64 resolution)
    model = create_model(num_frames=4, image_size=64, d_model=64, num_heads=4)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def predict_video(video_path, preview_save_dir=None, checkpoint_path=DEFAULT_CHECKPOINT_PATH, device='cpu'):
    """
    Runs full end-to-end inference on a single video file.
    
    Args:
        video_path: path to MP4/AVI/MOV video file
        preview_save_dir: directory where frame preview montage should be saved
        checkpoint_path: path to trained model weights
        device: 'cpu'
        
    Returns:
        dict containing prediction results, confidence scores, processing metrics, and preview image
    """
    if not is_model_trained(checkpoint_path):
        return {
            "status": "not_trained",
            "error": "The AI model is not trained yet. Please ask an Admin to train the model from the Model Training page."
        }

    start_time = time.time()

    try:
        # Step 1-6: Extract frames, detect faces, resize to 64x64, normalize
        tensor_frames, metadata, preview_filename = extract_and_preprocess_frames(
            video_path=video_path,
            num_frames=4,
            target_size=(64, 64),
            preview_save_dir=preview_save_dir
        )
    except Exception as e:
        return {
            "status": "error",
            "error": f"Video frame extraction failed: {str(e)}"
        }

    try:
        # Load model onto CPU
        model = load_model(checkpoint_path=checkpoint_path, device=device)

        # Step 7-11: Pass frames through CNN, Positional Encoding, Long-Distance Attention & Softmax
        with torch.no_grad():
            tensor_frames = tensor_frames.to(device)
            logits, attn_weights = model(tensor_frames, return_attention=True)
            probabilities = F.softmax(logits, dim=1)[0]

        # Class 0 = REAL, Class 1 = FAKE
        real_prob = float(probabilities[0].item())
        fake_prob = float(probabilities[1].item())

        real_percent = round(real_prob * 100, 2)
        fake_percent = round(fake_prob * 100, 2)

        # Determine label based on dominant probability
        prediction_label = "REAL" if real_percent >= fake_percent else "FAKE"
        processing_time = round(time.time() - start_time, 2)

        return {
            "status": "success",
            "prediction_label": prediction_label,
            "real_confidence": real_percent,
            "fake_confidence": fake_percent,
            "processing_time": processing_time,
            "frames_analyzed": metadata["frames_analyzed"],
            "faces_detected": metadata["faces_detected_count"],
            "preview_image": preview_filename,
            "metadata": metadata,
            "model_architecture": "CNN + Long-Distance Multi-Head Attention",
            "device": str(device).upper()
        }

    except Exception as e:
        return {
            "status": "error",
            "error": f"Prediction error: {str(e)}"
        }


if __name__ == "__main__":
    print(f"Is model trained: {is_model_trained()}")
