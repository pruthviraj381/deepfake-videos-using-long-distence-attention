import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from train import run_training_pipeline, BEST_MODEL_PATH

print("=== Starting Long-Distance Attention Training Test ===")
run_training_pipeline(epochs=3, batch_size=2, lr=0.0003, admin_id=1)
print("=== Training Complete ===")
print(f"Checkpoint exists: {os.path.exists(BEST_MODEL_PATH)}")
if os.path.exists(BEST_MODEL_PATH):
    print(f"Checkpoint size: {os.path.getsize(BEST_MODEL_PATH)} bytes")
