# Deepfake Video Detection Using Long-Distance Attention
A complete beginner-friendly MCA project with dataset preparation, model training, evaluation, and a Flask upload interface.

## Important
This ZIP contains the complete source code and training pipeline. It does **not** contain copyrighted/large real and fake video datasets or a pre-trained detector. You must add labelled training videos and run training before the upload page can make predictions.

Designed as a lightweight CPU-first educational baseline for a 4 GB RAM laptop. It uses a small CNN plus temporal self-attention over video frames. It is not a forensic-grade detector; results depend strongly on dataset quality and size.

## 1. Requirements
- Windows 10/11
- Python 3.10–3.13
- 4 GB RAM (close Chrome/other heavy apps while training)
- Disk space for videos and extracted cache

## 2. Extract the ZIP
Extract this folder somewhere simple, for example:
`C:\Deepfake_Detection_A_to_Z`

Open that folder in VS Code.

## 3. Create virtual environment
Open VS Code Terminal and run:
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PyTorch is already installed in your environment and works, you can keep it. Otherwise install CPU PyTorch using the official selector at https://pytorch.org/get-started/locally/ and then run `pip install -r requirements.txt`.

## 4. Add your labelled videos
Create this exact structure:
```text
dataset/
  train/
    real/
      real_video_01.mp4
      real_video_02.mp4
    fake/
      fake_video_01.mp4
      fake_video_02.mp4
  val/
    real/
    fake/
```
Put genuine videos only in `real`, manipulated/deepfake videos only in `fake`.
Use a separate set of videos in `val` (do not copy the same training videos into validation).
Supported: mp4, avi, mov, mkv.
Start with a small balanced set (for example 10–20 videos per class) to test the pipeline. More varied data is needed for meaningful performance.

## 5. Train the model
```bat
python train.py --data_dir dataset --epochs 5 --batch_size 1
```
For a 4 GB RAM laptop, begin with 3–5 epochs and batch size 1. Training can take a long time on CPU.
The script extracts 8 frames per video, resizes them to 96x96, trains the CNN + temporal attention model, prints validation accuracy, and saves:
`checkpoints/best_model.pth`

Optional:
```bat
python train.py --data_dir dataset --epochs 10 --batch_size 1 --frames 8 --image_size 96
```

## 6. Evaluate
Validation accuracy is printed after each epoch. Keep the validation videos separate from training videos. Accuracy on a tiny or biased dataset is not reliable.

## 7. Run the web application
After training successfully:
```bat
python app.py
```
Open the local address shown in the terminal (normally `http://127.0.0.1:5000`), upload an MP4 video, and click Analyze.
The app needs `checkpoints/best_model.pth`. If it is missing, train first.

## Project structure
```text
Deepfake_Detection_A_to_Z/
  app.py                 Flask upload and prediction application
  train.py               Complete training and validation script
  model.py               CNN + temporal long-distance attention model
  video_utils.py         Video frame extraction and preprocessing
  requirements.txt
  dataset/               Add your labelled videos here
  checkpoints/            Trained model is saved here
  templates/index.html    Frontend
  static/style.css        Frontend styling
```

## Troubleshooting
- `No module named torch`: activate `.venv`, then install CPU PyTorch.
- PyTorch DLL error on Windows: use a fresh Python 3.11/3.12 64-bit environment and install CPU PyTorch from the official PyTorch instructions. Avoid mixing packages between environments.
- `No videos found`: check the folder names and extensions exactly.
- Out of memory: close other apps; use batch size 1, 4 frames, and 80x80 images.
- Prediction says model missing: complete training first.
- If OpenCV cannot open a video, try converting it to a standard MP4 (H.264).

## Academic note
The model demonstrates a compact CNN feature extractor followed by temporal self-attention, allowing frame features to interact across the sampled sequence. It is a practical student project baseline, not proof that any video is authentic or manipulated. Include dataset details, split strategy, training settings, and measured results in your report.
