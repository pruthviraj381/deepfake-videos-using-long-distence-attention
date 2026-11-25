"""
End-to-End Functional Test for Flask Web Application
Tests:
- Admin login & Admin Dashboard
- User Registration & User Login
- Video Upload and Prediction Pipeline via Web
- Prediction Result Page Rendering
- History Page
- Admin Training API endpoints
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app import app
import database


class TestWebE2E(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_01_admin_login_and_dashboard(self):
        # Admin login
        resp = self.client.post('/admin/login', data={
            'username_or_email': 'admin',
            'password': 'admin123'
        }, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Administrator Dashboard', resp.data)
        self.assertIn(b'AI Model Training Status', resp.data)

        # Access Admin Training Page
        resp_train = self.client.get('/admin/training')
        self.assertEqual(resp_train.status_code, 200)
        self.assertIn(b'Model Training & Dataset Management', resp_train.data)

        # Access User Management Page
        resp_users = self.client.get('/admin/users')
        self.assertEqual(resp_users.status_code, 200)
        self.assertIn(b'Registered Users Directory', resp_users.data)

        # Access Training Status API
        resp_status = self.client.get('/admin/api/train/status')
        self.assertEqual(resp_status.status_code, 200)
        json_data = resp_status.get_json()
        self.assertIn('is_training', json_data)

    def test_02_user_flow_upload_and_prediction(self):
        # Register new user
        reg_resp = self.client.post('/register', data={
            'full_name': 'Alice Researcher',
            'username': 'alice_mca',
            'email': 'alice@university.edu',
            'password': 'password123',
            'confirm_password': 'password123'
        }, follow_redirects=True)
        self.assertEqual(reg_resp.status_code, 200)

        # Login as Alice
        login_resp = self.client.post('/login', data={
            'username_or_email': 'alice_mca',
            'password': 'password123'
        }, follow_redirects=True)
        self.assertEqual(login_resp.status_code, 200)
        self.assertIn(b'Welcome back, Alice Researcher', login_resp.data)

        # Upload a test video for deepfake analysis
        test_video_path = os.path.join(BASE_DIR, 'dataset', 'test', 'real', 'test_real_01.mp4')
        with open(test_video_path, 'rb') as video_fp:
            upload_resp = self.client.post('/upload', data={
                'input_type': 'file',
                'video_file': (video_fp, 'my_test_video.mp4')
            }, follow_redirects=True)

        self.assertEqual(upload_resp.status_code, 200)
        # Check that result page rendered with prediction verdict
        self.assertIn(b'Deepfake Detection Result', upload_resp.data)
        self.assertIn(b'PREDICTED AS', upload_resp.data)
        self.assertIn(b'REAL Confidence', upload_resp.data)
        self.assertIn(b'FAKE Confidence', upload_resp.data)

        # Check prediction history page
        hist_resp = self.client.get('/history')
        self.assertEqual(hist_resp.status_code, 200)
        self.assertIn(b'my_test_video.mp4', hist_resp.data)


if __name__ == '__main__':
    unittest.main()
