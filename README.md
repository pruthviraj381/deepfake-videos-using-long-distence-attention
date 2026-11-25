# Detection of Deepfake Videos Using Long-Distance Attention

**MCA Final-Year Academic Project**  
*A Full-Stack Deep Learning Web Application for Video Manipulation and Deepfake Detection*

---

## 1. Project Overview & Objective

With the rapid advancement of generative AI (Autoencoders, GANs, and Diffusion models), creating hyper-realistic deepfake videos has become increasingly accessible. Traditional deepfake detectors rely on single-frame CNNs that analyze spatial facial anomalies in isolation. However, high-quality modern deepfakes often produce imperceptible single-frame artifacts but exhibit **temporal flickering, boundary blending seams, unnatural eye-blinking cycles, and inconsistent pose transitions across distant frames**.

This project implements a novel **Long-Distance Attention Deepfake Detection System**:
1. Uniformly samples temporal frames spanning the entire video duration (capturing long-distance temporal transitions).
2. Uses OpenCV Haar Cascades to detect, crop, and normalize the facial region of interest to **64 &times; 64**.
3. Passes cropped frames into a lightweight **Convolutional Frame Encoder** to extract spatial facial features.
4. Injects **Temporal Positional Encoding** to preserve sequential and distance context.
5. Applies **Multi-Head Long-Distance Attention** to model correlations between distant frames.
6. Computes classification probabilities via **Softmax** into **REAL** (authentic) or **FAKE** (manipulated).
7. Engineered specifically to run seamlessly on **pure CPU execution with 4 GB RAM**.

---

## 2. Technologies Used

- **Deep Learning Framework:** PyTorch 2.x (CPU execution)
- **Computer Vision:** OpenCV (`cv2`), NumPy, Pillow (`PIL`)
- **Backend Web Framework:** Python 3, Flask 3, Werkzeug (Password Hashing & Secure Filenames)
- **Database:** SQLite3 (`deepfake_detection.db`)
- **Video Download & Streaming:** `requests`, `urllib`, `yt-dlp`
- **Frontend & UI/UX:** HTML5, CSS3, JavaScript, Bootstrap 5, Bootstrap Icons, Google Fonts
- **Hardware Profile:** 4 GB RAM, No GPU required (CPU-optimized)

---

## 3. Architecture & Mathematical Principles

```
Video Input (MP4/AVI/MOV / URL)
       │
       ▼
Uniform Long-Distance Frame Sampling (T = 4 frames)
       │
       ▼
OpenCV Haar Cascade Face Detection & Margin Crop
       │
       ▼
Resize (64 × 64) & ImageNet Normalization -> Shape: [B, 4, 3, 64, 64]
       │
       ▼
CNN Spatial Frame Encoder (4 Conv2D + BatchNorm + MaxPool blocks)
       │
       ▼
Linear Projection to d_model (64) -> Shape: [B, 4, 64]
       │
       ▼
Learnable Temporal Positional Encoding (Added element-wise)
       │
       ▼
Multi-Head Long-Distance Attention (num_heads = 4)
  Attention(Q, K, V) = softmax(Q · K^T / sqrt(d_k)) · V
       │
       ▼
Residual Connections + Layer Normalization + Feed-Forward Network
       │
       ▼
Temporal Global Average Pooling -> Shape: [B, 64]
       │
       ▼
Classification Head (Linear -> LayerNorm -> ReLU -> Dropout -> Linear)
       │
       ▼
Softmax Activation -> Probabilities: P(REAL) vs P(FAKE)
```

---

## 4. Project Folder Structure

```
deepfake_detection/
│
├── app.py                      # Flask web application & API routing
├── model.py                    # PyTorch Long-Distance Attention Neural Network
├── train.py                    # PyTorch training pipeline & background worker
├── predict.py                  # End-to-end video inference & prediction engine
├── video_utils.py              # OpenCV face detection, frame sampling & URL downloader
├── database.py                 # SQLite database layer, schema & CRUD functions
├── generate_sample_dataset.py  # Benchmark dataset video generator
├── check_env.py                # Environment verification script
├── requirements.txt            # Python dependencies
├── README.md                   # Complete academic documentation
├── deepfake_detection.db       # SQLite local database (auto-generated)
│
├── dataset/                    # Isolated dataset splits (prevent data leakage)
│   ├── train/
│   │   ├── real/               # Authentic training video clips
│   │   └── fake/               # Deepfake training video clips
│   ├── val/
│   │   ├── real/               # Authentic validation clips
│   │   └── fake/               # Deepfake validation clips
│   └── test/
│       ├── real/               # Authentic test clips
│       └── fake/               # Deepfake test clips
│
├── checkpoints/
│   └── best_model.pth          # Saved PyTorch model checkpoint (weights & metadata)
│
├── templates/                  # Responsive Bootstrap 5 HTML templates
│   ├── base.html               # Master layout with sidebar & topbar
│   ├── index.html              # Landing page with architecture & statistics
│   ├── login.html              # Unified User & Admin login portal
│   ├── register.html           # User registration page
│   ├── user_dashboard.html     # User workspace & quick action cards
│   ├── admin_dashboard.html    # Admin analytics & dataset breakdown
│   ├── upload.html             # Video drag-and-drop & online URL detection
│   ├── result.html             # Detailed prediction result & confidence gauges
│   ├── training.html           # Admin model training console & live progress
│   ├── history.html            # System prediction history log
│   └── users.html              # Admin user management & access controls
│
├── static/
│   ├── css/
│   │   └── style.css           # Modern dark blue & crisp white styling
│   ├── js/
│   │   └── script.js           # Drag-drop upload, live polling & animations
│   ├── previews/               # Auto-saved 4-frame facial montage images
│   └── images/
│
└── uploads/                    # User uploaded videos and downloaded online videos
```

---

## 5. 4GB RAM Optimization Strategies

To operate stably within a **4 GB RAM budget without GPU hardware**:
1. **CPU Tensor Processing:** PyTorch operates in CPU mode with `torch.no_grad()` during inference, keeping peak RAM usage below 350 MB.
2. **Sparse Long-Distance Sampling:** Rather than processing hundreds of consecutive frames (which crashes low-RAM systems), the model uniformly samples **4 key temporal frames** ($T=4$) spanning beginning, middle, and end.
3. **64 &times; 64 Normalized Resolution:** Face crops are resized to $64 \times 64$, ensuring matrix multiplications take milliseconds on CPU.
4. **Mini-Batch Training:** Training runs with `batch_size=2` (or 4) with zero multithreaded DataLoader overhead (`num_workers=0`), preventing Windows paging exhaustion.
5. **Garbage Collection & Video Release:** OpenCV VideoCapture objects and PyTorch tensors are explicitly released after each operation.

---

## 6. Installation & Execution Guide (Windows Command Prompt)

### Step 1: Open Windows Command Prompt
Open `cmd.exe` or PowerShell and navigate to the project folder:
```cmd
cd c:\Users\RRC\Desktop\project\deepfake_detection
```

### Step 2: Install Python Dependencies
Install all required libraries using pip:
```cmd
pip install -r requirements.txt
```

### Step 3: Generate Benchmark Sample Dataset
Generate benchmark video clips (real human facial motion vs deepfake seam manipulation) to populate `dataset/train`, `dataset/val`, and `dataset/test`:
```cmd
python generate_sample_dataset.py
```

### Step 4: Train the AI Model (Initial Run)
You can train the model from the CLI or via the Admin Web UI. To train directly via CLI:
```cmd
python train.py
```
This trains the model on the benchmark dataset on CPU and creates `checkpoints/best_model.pth`.

### Step 5: Start the Flask Web Application
```cmd
python app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 7. Default Credentials & User Roles

### Administrator Portal
- **URL:** `http://127.0.0.1:5000/admin/login`
- **Username:** `admin`
- **Password:** `admin123`
- **Capabilities:**
  - View overall system statistics (users, videos, real/fake count).
  - Train the Long-Distance Attention model with custom epochs and batch sizes.
  - Monitor real-time training progress, loss, and accuracy curves.
  - Upload real and fake videos into `train/`, `val/`, or `test/` dataset folders.
  - Manage user accounts (suspend, activate, or delete).
  - Inspect all system-wide prediction records.

### User Portal
- **URL:** `http://127.0.0.1:5000/register` or `http://127.0.0.1:5000/login`
- **Capabilities:**
  - Drag-and-drop video upload (MP4, AVI, MOV).
  - Online video URL detection (Direct HTTP/HTTPS MP4 links or platform links).
  - View real-time prediction results (REAL in green, FAKE in red).
  - View confidence percentages, processing time, and 4-frame facial breakdown.
  - View personal prediction history.

---

## 8. Academic Viva Questions & Model Defense

**Q1: Why use Long-Distance Attention instead of a standard 3D CNN or LSTM?**  
*Answer:* Standard 3D CNNs and LSTMs suffer from vanishing gradients when modeling long sequences and incur heavy computational memory costs. Long-Distance Multi-Head Attention computes direct pairwise temporal correlations between distant frames in a single matrix operation, efficiently identifying subtle temporal boundary shifts and warping artifacts without GPU requirements.

**Q2: How does the model prevent data leakage during training?**  
*Answer:* Video samples are strictly separated into isolated directories: `dataset/train`, `dataset/val`, and `dataset/test`. Frames from the same video are never split across both training and validation sets.

**Q3: How is face detection handled if a video has no clear face?**  
*Answer:* OpenCV's Haar Cascade classifier (`haarcascade_frontalface_default.xml`) searches for frontal faces. If found, it adds a 15% margin around the bounding box. If no face is detected (e.g. extreme angle or profile), a centered square crop is used as a fallback to ensure continuous processing without crashing.

**Q4: How are confidence scores calculated?**  
*Answer:* The final classification layer outputs 2 raw logits for $[REAL, FAKE]$. Applying the Softmax function:
$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{2} e^{z_j}}$$
yields valid probabilities summing to 1.0 (converted to percentages).
