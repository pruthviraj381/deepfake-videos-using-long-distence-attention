
from pathlib import Path
import uuid
import os
import torch

from flask import (
    Flask, render_template, request,
    redirect, url_for, session
)
from werkzeug.utils import secure_filename
from functools import wraps
from datetime import datetime

from model import DeepfakeAttentionModel
from video_utils import read_video_frames, VIDEO_EXTENSIONS

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY", "deepfake-project-local-secret"
)

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
UPLOAD = BASE_DIR / "uploads"
UPLOAD.mkdir(exist_ok=True)

CHECKPOINT = BASE_DIR / "checkpoints" / "groupnorm_best_model.pth"

model = None
model_settings = {"frames": 4, "image_size": 64}

USERNAME = "admin"
PASSWORD = "admin123"

history = []


# LOAD TRAINED MODEL
def load_detector():
    global model, model_settings

    if model is not None:
        return True

    if not CHECKPOINT.exists():
        print("Checkpoint not found:", CHECKPOINT)
        return False

    data = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    if data.get("architecture") != "GroupNorm":
        raise ValueError("Incorrect model checkpoint architecture.")

    model_settings = {
        "frames": data.get("frames", 4),
        "image_size": data.get("image_size", 64)
    }

    model = DeepfakeAttentionModel()

    model.load_state_dict(
        data["model_state_dict"],
        strict=True
    )

    model.eval()

    print("GroupNorm model loaded successfully!")

    return True


# LOGIN PROTECTION
def login_required(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return function(*args, **kwargs)
    return wrapper


# LOGIN
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():

    if "username" in session:
        return redirect(url_for("dashboard"))

    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if username == USERNAME and password == PASSWORD:
            session.clear()
            session["username"] = username
            return redirect(url_for("dashboard"))

        error = "Invalid username or password!"

    return render_template("login.html", error=error)


# DASHBOARD AND VIDEO DETECTION
@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():

    result = None
    error = None
    ready = False

    try:
        ready = load_detector()
    except Exception as e:
        error = f"Model loading failed: {e}"

    if request.method == "POST":

        if not ready:
            error = error or "Trained GroupNorm model not found."

        elif "video" not in request.files:
            error = "Please select a video."

        else:
            f = request.files["video"]

            if not f.filename:
                error = "Please select a video."

            else:
                safe = secure_filename(f.filename)

                if Path(safe).suffix.lower() not in VIDEO_EXTENSIONS:
                    error = "Unsupported video format."

                else:
                    path = UPLOAD / f"{uuid.uuid4().hex}_{safe}"
                    f.save(path)

                    try:
                        x = read_video_frames(
                            path,
                            model_settings["frames"],
                            model_settings["image_size"]
                        ).unsqueeze(0)

                        with torch.no_grad():
                            probs = torch.softmax(model(x), dim=1)[0]

                        real_percentage = float(probs[0]) * 100
                        fake_percentage = float(probs[1]) * 100

                        label = "FAKE" if fake_percentage > real_percentage else "REAL"

                        result = {
                            "filename": safe,
                            "label": label,
                            "real": round(real_percentage, 2),
                            "fake": round(fake_percentage, 2),
                            "confidence": round(max(real_percentage, fake_percentage), 2)
                        }

                        history.insert(0, {
                            **result,
                            "date": datetime.now().strftime("%d-%m-%Y %H:%M")
                        })

                        history[:] = history[:20]

                        del x, probs

                    except Exception as e:
                        error = f"Could not analyze video: {e}"

                    finally:
                        path.unlink(missing_ok=True)

    return render_template(
        "dashboard.html",
        ready=ready,
        result=result,
        error=error,
        history=history,
        username=session["username"]
    )


# LOGOUT
@app.route("/logout")
@login_required
def logout():
    session.clear()
    return render_template("logout.html")


# UPLOAD SIZE ERROR
@app.errorhandler(413)
def file_too_large(error):
    return "Video too large. Maximum upload size is 100 MB.", 413


if __name__ == "__main__":
    torch.set_num_threads(2)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
