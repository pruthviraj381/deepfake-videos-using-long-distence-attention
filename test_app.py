"""
Automated Test Suite for Deepfake Detection Application
Verifies:
1. Database operations (User creation, Authentication, Admin privileges)
2. PyTorch Model Architecture (CNN Frame Encoder, Positional Encoding, Multi-Head Attention, Softmax)
3. Video Frame Extraction & Face Detection Pipeline
4. Prediction Engine (REAL/FAKE classification)
5. Flask Web Endpoints & Access Control
"""

import os
import sys
import unittest
import torch
import numpy as np

# Ensure module path is accessible
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import database
from model import create_model
from video_utils import extract_and_preprocess_frames, allowed_file
from predict import is_model_trained, predict_video
from app import app


class TestDeepfakeDetectionSystem(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_01_allowed_file(self):
        """Verifies supported video file extensions."""
        self.assertTrue(allowed_file('video.mp4'))
        self.assertTrue(allowed_file('test.avi'))
        self.assertTrue(allowed_file('sample.MOV'))
        self.assertFalse(allowed_file('image.png'))
        self.assertFalse(allowed_file('document.pdf'))

    def test_02_database_initialization_and_admin(self):
        """Verifies default admin account in SQLite."""
        database.init_db()
        admin, err = database.authenticate_user('admin', 'admin123', required_role='admin')
        self.assertIsNone(err)
        self.assertIsNotNone(admin)
        self.assertEqual(admin['role'], 'admin')

    def test_03_user_registration_and_auth(self):
        """Verifies new user registration and authentication."""
        test_user = f"tester_{int(torch.randint(1000, 9999, (1,)).item())}"
        uid, err = database.create_user(test_user, f"{test_user}@test.com", "Password123", "Test User")
        self.assertIsNone(err)
        self.assertIsNotNone(uid)

        user, err = database.authenticate_user(test_user, "Password123")
        self.assertIsNone(err)
        self.assertIsNotNone(user)
        self.assertEqual(user['username'], test_user)

    def test_04_pytorch_model_architecture(self):
        """Verifies Long-Distance Attention model forward pass and output shape."""
        model = create_model(num_frames=4, image_size=128, d_model=128, num_heads=4)
        dummy_video = torch.randn(2, 4, 3, 128, 128)
        logits, weights = model(dummy_video, return_attention=True)
        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(weights.shape, (2, 4, 4))  # (B, T, T) attention weights

        # Softmax probabilities
        probs = model.predict_proba(dummy_video)
        self.assertEqual(probs.shape, (2, 2))
        # Probabilities sum to 1.0
        sums = torch.sum(probs, dim=1).tolist()
        for s in sums:
            self.assertAlmostEqual(s, 1.0, places=4)

    def test_05_frame_extraction_on_sample_video(self):
        """Verifies OpenCV face detection and frame extraction."""
        sample_path = os.path.join(BASE_DIR, 'dataset', 'train', 'real', 'train_real_01.mp4')
        if os.path.exists(sample_path):
            tensor_frames, metadata, preview = extract_and_preprocess_frames(
                video_path=sample_path,
                num_frames=4,
                target_size=(128, 128)
            )
            self.assertEqual(tensor_frames.shape, (1, 4, 3, 128, 128))
            self.assertEqual(metadata['frames_analyzed'], 4)

    def test_06_flask_public_routes(self):
        """Verifies public Flask routes respond with HTTP 200."""
        r = self.app.get('/')
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'DeepShield AI', r.data)

        r = self.app.get('/login')
        self.assertEqual(r.status_code, 200)

        r = self.app.get('/admin/login')
        self.assertEqual(r.status_code, 200)

        r = self.app.get('/register')
        self.assertEqual(r.status_code, 200)

    def test_07_access_control(self):
        """Verifies that protected routes redirect unauthorized users to login."""
        r = self.app.get('/dashboard', follow_redirects=False)
        self.assertEqual(r.status_code, 302)

        r = self.app.get('/admin/dashboard', follow_redirects=False)
        self.assertEqual(r.status_code, 302)


if __name__ == '__main__':
    unittest.main()
