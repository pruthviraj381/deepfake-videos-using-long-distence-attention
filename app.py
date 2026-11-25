"""
Deepfake Video Detection Web Application using Long-Distance Attention
Flask Backend Server
MCA Final-Year Academic Project
"""

import os
import shutil
import time
from functools import wraps
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify, send_from_directory, abort
)
from werkzeug.utils import secure_filename

import database
from video_utils import (
    allowed_file, get_video_metadata,
    ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES
)
from predict import predict_video, is_model_trained, DEFAULT_CHECKPOINT_PATH, TORCH_ERROR
from train import (
    start_training_background, get_training_status, get_dataset_counts,
    DATASET_DIR, BEST_MODEL_PATH
)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'deepfake_long_distance_attention_mca_secret_key_2026')
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_BYTES

@app.context_processor
def inject_torch_status():
    return {
        'torch_error': TORCH_ERROR,
        'has_torch': TORCH_ERROR is None
    }

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PREVIEW_FOLDER = os.path.join(BASE_DIR, 'static', 'previews')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREVIEW_FOLDER, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Initialize SQLite database
database.init_db()


# ---------------- AUTH DECORATORS ---------------- #

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'admin':
            flash("Administrator privileges required to access this page.", "danger")
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function


# ---------------- GENERAL & AUTH ROUTES ---------------- #

@app.route('/')
def index():
    """Landing homepage presenting the project, architecture and entry points."""
    model_trained = is_model_trained()
    stats = database.get_system_statistics()
    return render_template('index.html', model_trained=model_trained, stats=stats)


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page."""
    if 'user_id' in session:
        return redirect(url_for('admin_dashboard' if session.get('role') == 'admin' else 'user_dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        full_name = request.form.get('full_name', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not username or not email or not password:
            flash("Please fill in all required fields.", "danger")
            return render_template('register.html')

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "danger")
            return render_template('register.html')

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template('register.html')

        user_id, error = database.create_user(username, email, password, full_name=full_name, role='user')
        if error:
            flash(error, "danger")
            return render_template('register.html')

        flash("Account created successfully! You can now log in.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Unified user & admin login."""
    if 'user_id' in session:
        return redirect(url_for('admin_dashboard' if session.get('role') == 'admin' else 'user_dashboard'))

    if request.method == 'POST':
        username_or_email = request.form.get('username_or_email', '').strip()
        password = request.form.get('password', '')

        user, error = database.authenticate_user(username_or_email, password)
        if error:
            flash(error, "danger")
            return render_template('login.html', is_admin=False)

        session['user_id'] = user['id']
        session['username'] = user['username']
        session['email'] = user['email']
        session['role'] = user['role']
        session['full_name'] = user['full_name'] or user['username']

        flash(f"Welcome back, {session['full_name']}!", "success")
        next_page = request.args.get('next')
        if next_page:
            return redirect(next_page)

        return redirect(url_for('admin_dashboard' if user['role'] == 'admin' else 'user_dashboard'))

    return render_template('login.html', is_admin=False)


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Dedicated Administrator login."""
    if 'user_id' in session and session.get('role') == 'admin':
        return redirect(url_for('admin_dashboard'))

    if request.method == 'POST':
        username_or_email = request.form.get('username_or_email', '').strip()
        password = request.form.get('password', '')

        user, error = database.authenticate_user(username_or_email, password, required_role='admin')
        if error:
            flash(error, "danger")
            return render_template('login.html', is_admin=True)

        session['user_id'] = user['id']
        session['username'] = user['username']
        session['email'] = user['email']
        session['role'] = 'admin'
        session['full_name'] = user['full_name'] or user['username']

        flash("Administrator authenticated successfully.", "success")
        return redirect(url_for('admin_dashboard'))

    return render_template('login.html', is_admin=True)


@app.route('/logout')
def logout():
    """Terminates active session."""
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))


# ---------------- USER MODULE ROUTES ---------------- #

@app.route('/dashboard')
@login_required
def user_dashboard():
    """User welcome dashboard displaying personal prediction statistics and quick actions."""
    user_id = session['user_id']
    stats = database.get_user_statistics(user_id)
    recent_predictions = database.get_user_predictions(user_id, limit=5)
    model_trained = is_model_trained()
    return render_template(
        'user_dashboard.html',
        stats=stats,
        recent_predictions=recent_predictions,
        model_trained=model_trained
    )


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_video():
    """Upload video file for deepfake analysis using Long-Distance Attention."""
    model_trained = is_model_trained()

    if request.method == 'POST':
        user_id = session['user_id']

        # File upload validation
        if 'video_file' not in request.files:
            flash("No video file selected.", "danger")
            return render_template('upload.html', model_trained=model_trained)

        file = request.files['video_file']
        if file.filename == '':
            flash("No video file selected.", "danger")
            return render_template('upload.html', model_trained=model_trained)

        if not allowed_file(file.filename):
            flash("Invalid format. Only MP4, AVI, and MOV video files are supported.", "danger")
            return render_template('upload.html', model_trained=model_trained)

        original_name = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        save_name = f"user_{user_id}_{timestamp}_{original_name}"
        saved_filepath = os.path.join(UPLOAD_FOLDER, save_name)
        file.save(saved_filepath)

        # Validate video using OpenCV
        file_size = os.path.getsize(saved_filepath)
        video_id = database.save_video_record(
            user_id=user_id,
            filename=os.path.basename(saved_filepath),
            original_name=original_name,
            filepath=saved_filepath,
            file_size=file_size,
            source_type="upload",
            source_url=None
        )

        # If model is not trained, notify and redirect
        if not model_trained:
            flash(
                "Video uploaded successfully. However, the AI model has not been trained yet. "
                "Predictions cannot be run until an Administrator trains the model.",
                "warning"
            )
            return redirect(url_for('user_dashboard'))

        # Run prediction through Long-Distance Attention model
        result = predict_video(
            video_path=saved_filepath,
            preview_save_dir=PREVIEW_FOLDER,
            checkpoint_path=DEFAULT_CHECKPOINT_PATH
        )

        if result.get("status") != "success":
            flash(f"Analysis error: {result.get('error', 'Unknown error')}", "danger")
            return redirect(url_for('upload_video'))

        # Save prediction into database
        prediction_id = database.save_prediction(
            video_id=video_id,
            user_id=user_id,
            prediction_label=result["prediction_label"],
            real_confidence=result["real_confidence"],
            fake_confidence=result["fake_confidence"],
            frames_analyzed=result["frames_analyzed"],
            processing_time=result["processing_time"],
            preview_image=result["preview_image"],
            model_version="LongDistanceAttention-v1"
        )

        flash("Video analysis completed successfully!", "success")
        return redirect(url_for('prediction_result', prediction_id=prediction_id))

    return render_template('upload.html', model_trained=model_trained)


@app.route('/result/<int:prediction_id>')
@login_required
def prediction_result(prediction_id):
    """Detailed prediction result page showing REAL/FAKE status, confidence gauges, and frame montage."""
    prediction = database.get_prediction_by_id(prediction_id)
    if not prediction:
        flash("Prediction record not found.", "danger")
        return redirect(url_for('user_dashboard'))

    # Access control: users can only see their own predictions unless admin
    if session.get('role') != 'admin' and prediction['user_id'] != session['user_id']:
        abort(403)

    return render_template('result.html', prediction=prediction)


@app.route('/history')
@login_required
def prediction_history():
    """Prediction history page for logged in user."""
    user_id = session['user_id']
    history = database.get_user_predictions(user_id, limit=100)
    return render_template('history.html', history=history, is_admin=False)


# ---------------- ADMIN MODULE ROUTES ---------------- #

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    """Admin dashboard with high-level statistics, system metrics, and dataset overview."""
    stats = database.get_system_statistics()
    dataset_counts = get_dataset_counts()
    recent_training = database.get_latest_training_stats()
    all_recent = database.get_all_predictions(limit=10)
    model_trained = is_model_trained()

    return render_template(
        'admin_dashboard.html',
        stats=stats,
        dataset_counts=dataset_counts,
        recent_training=recent_training,
        all_recent=all_recent,
        model_trained=model_trained
    )


@app.route('/admin/training')
@admin_required
def admin_training():
    """Model training page with dataset counters, progress monitor, and training controls."""
    dataset_counts = get_dataset_counts()
    history = database.get_training_history(limit=10)
    model_trained = is_model_trained()
    status = get_training_status()

    return render_template(
        'training.html',
        dataset_counts=dataset_counts,
        history=history,
        model_trained=model_trained,
        status=status
    )


@app.route('/admin/api/train/start', methods=['POST'])
@admin_required
def api_train_start():
    """API endpoint to trigger model training in background thread."""
    try:
        epochs = int(request.form.get('epochs', 5))
        batch_size = int(request.form.get('batch_size', 2))
        lr = float(request.form.get('lr', 0.0003))
    except ValueError:
        return jsonify({"success": False, "message": "Invalid training parameters."}), 400

    admin_id = session['user_id']
    success, message = start_training_background(epochs=epochs, batch_size=batch_size, lr=lr, admin_id=admin_id)
    return jsonify({"success": success, "message": message})


@app.route('/admin/api/train/status')
@admin_required
def api_train_status():
    """API endpoint to poll live training status and metrics."""
    status = get_training_status()
    return jsonify(status)


@app.route('/admin/dataset/upload', methods=['POST'])
@admin_required
def admin_dataset_upload():
    """Allows Admin to upload REAL and FAKE videos into dataset folders."""
    category = request.form.get('category', 'real').lower()
    split = request.form.get('split', 'train').lower()

    if category not in ['real', 'fake'] or split not in ['train', 'val', 'test']:
        flash("Invalid dataset category or split specified.", "danger")
        return redirect(url_for('admin_training'))

    if 'video_file' not in request.files:
        flash("No video file selected for dataset upload.", "danger")
        return redirect(url_for('admin_training'))

    file = request.files['video_file']
    if file.filename == '':
        flash("No video file selected.", "danger")
        return redirect(url_for('admin_training'))

    if not allowed_file(file.filename):
        flash("Invalid file format. Please upload MP4, AVI, or MOV video.", "danger")
        return redirect(url_for('admin_training'))

    filename = secure_filename(file.filename)
    dest_dir = os.path.join(DATASET_DIR, split, category)
    os.makedirs(dest_dir, exist_ok=True)

    dest_path = os.path.join(dest_dir, f"{int(time.time())}_{filename}")
    file.save(dest_path)

    flash(f"Successfully added video to {split.upper()} / {category.upper()} dataset!", "success")
    return redirect(url_for('admin_training'))


@app.route('/admin/users')
@admin_required
def admin_users():
    """User management page listing registered users and their activity."""
    users = database.get_all_users()
    return render_template('users.html', users=users)


@app.route('/admin/users/toggle/<int:user_id>', methods=['POST'])
@admin_required
def admin_toggle_user(user_id):
    """Toggles user active / suspended status."""
    success, msg = database.toggle_user_status(user_id)
    flash(msg, "success" if success else "danger")
    return redirect(url_for('admin_users'))


@app.route('/admin/users/delete/<int:user_id>', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    """Deletes a user account."""
    success, msg = database.delete_user(user_id)
    flash(msg, "success" if success else "danger")
    return redirect(url_for('admin_users'))


@app.route('/admin/history')
@admin_required
def admin_history():
    """System-wide prediction history table for administrators."""
    history = database.get_all_predictions(limit=200)
    return render_template('history.html', history=history, is_admin=True)


# ---------------- STATIC SERVING ---------------- #

@app.route('/uploads/<filename>')
@login_required
def serve_upload(filename):
    """Safely serves uploaded video files."""
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/static/previews/<filename>')
def serve_preview(filename):
    """Serves generated facial preview montage images."""
    return send_from_directory(PREVIEW_FOLDER, filename)


# ---------------- ERROR HANDLERS ---------------- #

@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html', error_msg="The requested page could not be found."), 404


@app.errorhandler(413)
def request_entity_too_large(error):
    flash("The uploaded file exceeds the 50 MB size limit.", "danger")
    return redirect(request.url), 413


@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html', error_msg="An internal server error occurred."), 500


if __name__ == '__main__':
    print("=" * 65)
    print("  Deepfake Video Detection using Long-Distance Attention")
    print("  MCA Final Year Academic Project")
    print("  Starting Flask Web Application on http://127.0.0.1:5000")
    print("=" * 65)
    app.run(host='0.0.0.0', port=5000, debug=True)
