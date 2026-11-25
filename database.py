"""
Database Management Layer for Deepfake Video Detection System
Uses SQLite3 for lightweight, zero-configuration local storage.
Tables:
- users
- admin
- videos
- predictions
- training_history
"""

import os
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deepfake_detection.db')


def get_db_connection():
    """Returns a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes SQLite database tables and default admin account if not present."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT,
        role TEXT DEFAULT 'user',
        status TEXT DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 2. Admin table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS admin (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER UNIQUE,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        last_login TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """)

    # 3. Videos table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS videos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        filename TEXT NOT NULL,
        original_name TEXT NOT NULL,
        filepath TEXT NOT NULL,
        file_size INTEGER DEFAULT 0,
        source_type TEXT DEFAULT 'upload',
        source_url TEXT,
        duration REAL DEFAULT 0.0,
        fps REAL DEFAULT 0.0,
        frame_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
    )
    """)

    # 4. Predictions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id INTEGER NOT NULL,
        user_id INTEGER,
        prediction_label TEXT NOT NULL,
        real_confidence REAL NOT NULL,
        fake_confidence REAL NOT NULL,
        frames_analyzed INTEGER DEFAULT 4,
        processing_time REAL DEFAULT 0.0,
        preview_image TEXT,
        model_version TEXT DEFAULT 'LongDistanceAttention-v1',
        status TEXT DEFAULT 'completed',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
    )
    """)

    # 5. Training History table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS training_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        admin_id INTEGER,
        epochs INTEGER NOT NULL,
        batch_size INTEGER NOT NULL,
        train_loss REAL DEFAULT 0.0,
        train_accuracy REAL DEFAULT 0.0,
        val_loss REAL DEFAULT 0.0,
        val_accuracy REAL DEFAULT 0.0,
        checkpoint_path TEXT,
        dataset_real_count INTEGER DEFAULT 0,
        dataset_fake_count INTEGER DEFAULT 0,
        status TEXT DEFAULT 'completed',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (admin_id) REFERENCES users(id) ON DELETE SET NULL
    )
    """)

    conn.commit()

    # Seed default Admin account if not present
    cursor.execute("SELECT * FROM users WHERE role = 'admin' LIMIT 1")
    admin_user = cursor.fetchone()

    if not admin_user:
        admin_username = "admin"
        admin_email = "admin@deepfake.ai"
        admin_pwd = generate_password_hash("admin123")

        cursor.execute("""
            INSERT INTO users (username, email, password_hash, full_name, role, status)
            VALUES (?, ?, ?, ?, 'admin', 'active')
        """, (admin_username, admin_email, admin_pwd, "System Administrator"))
        admin_id = cursor.lastrowid

        cursor.execute("""
            INSERT INTO admin (user_id, username, email, password_hash)
            VALUES (?, ?, ?, ?)
        """, (admin_id, admin_username, admin_email, admin_pwd))

        conn.commit()
        print(f"[Database] Default Admin created: username='{admin_username}', password='admin123'")

    conn.close()


# ---------------- USER FUNCTIONS ---------------- #

def create_user(username, email, password, full_name="", role="user"):
    """Registers a new user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    pwd_hash = generate_password_hash(password)
    try:
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (?, ?, ?, ?, ?)
        """, (username.strip(), email.strip().lower(), pwd_hash, full_name.strip(), role))
        user_id = cursor.lastrowid

        if role == 'admin':
            cursor.execute("""
                INSERT INTO admin (user_id, username, email, password_hash)
                VALUES (?, ?, ?, ?)
            """, (user_id, username.strip(), email.strip().lower(), pwd_hash))

        conn.commit()
        return user_id, None
    except sqlite3.IntegrityError as e:
        err_msg = str(e).lower()
        if "username" in err_msg:
            return None, "Username is already taken."
        elif "email" in err_msg:
            return None, "Email is already registered."
        return None, "User with this username or email already exists."
    finally:
        conn.close()


def authenticate_user(username_or_email, password, required_role=None):
    """Authenticates user or admin. Returns user row dict or None."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM users 
        WHERE (LOWER(username) = LOWER(?) OR LOWER(email) = LOWER(?))
        AND status = 'active'
    """, (username_or_email.strip(), username_or_email.strip()))
    user = cursor.fetchone()

    if user and check_password_hash(user['password_hash'], password):
        if required_role and user['role'] != required_role:
            conn.close()
            return None, f"Unauthorized: {required_role.capitalize()} access required."

        # Update last login for admin if applicable
        if user['role'] == 'admin':
            cursor.execute("UPDATE admin SET last_login = ? WHERE user_id = ?",
                           (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user['id']))
            conn.commit()

        user_dict = dict(user)
        conn.close()
        return user_dict, None

    conn.close()
    return None, "Invalid username/email or password."


def get_user_by_id(user_id):
    """Fetches user by id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, full_name, role, status, created_at FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None


def get_all_users():
    """Returns list of all registered users with video count and prediction count."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            u.id, u.username, u.email, u.full_name, u.role, u.status, u.created_at,
            COUNT(DISTINCT v.id) as total_videos,
            COUNT(DISTINCT p.id) as total_predictions
        FROM users u
        LEFT JOIN videos v ON u.id = v.user_id
        LEFT JOIN predictions p ON u.id = p.user_id
        GROUP BY u.id
        ORDER BY u.created_at DESC
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def toggle_user_status(user_id):
    """Toggles user status between 'active' and 'suspended'."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT status, role FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    if not user:
        conn.close()
        return False, "User not found."
    if user['role'] == 'admin':
        conn.close()
        return False, "Cannot modify status of Admin account."

    new_status = 'suspended' if user['status'] == 'active' else 'active'
    cursor.execute("UPDATE users SET status = ? WHERE id = ?", (new_status, user_id))
    conn.commit()
    conn.close()
    return True, f"User status changed to {new_status}."


def delete_user(user_id):
    """Deletes a user (prevent deleting main admin)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT role FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    if not user:
        conn.close()
        return False, "User not found."
    if user['role'] == 'admin':
        conn.close()
        return False, "Cannot delete an Admin account."

    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return True, "User deleted successfully."


# ---------------- VIDEO & PREDICTION FUNCTIONS ---------------- #

def save_video_record(user_id, filename, original_name, filepath, file_size=0, source_type='upload', source_url=None, duration=0.0, fps=0.0, frame_count=0):
    """Inserts a new video record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO videos (user_id, filename, original_name, filepath, file_size, source_type, source_url, duration, fps, frame_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, filename, original_name, filepath, file_size, source_type, source_url, duration, fps, frame_count))
    video_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return video_id


def save_prediction(video_id, user_id, prediction_label, real_confidence, fake_confidence, frames_analyzed=4, processing_time=0.0, preview_image=None, model_version='LongDistanceAttention-v1'):
    """Inserts a prediction result."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (video_id, user_id, prediction_label, real_confidence, fake_confidence, frames_analyzed, processing_time, preview_image, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (video_id, user_id, prediction_label, real_confidence, fake_confidence, frames_analyzed, processing_time, preview_image, model_version))
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return prediction_id


def get_prediction_by_id(prediction_id):
    """Fetches single prediction with associated video and user details."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            p.*, 
            v.filename, v.original_name, v.filepath, v.file_size, v.source_type, v.source_url, v.duration, v.fps,
            u.username, u.email
        FROM predictions p
        JOIN videos v ON p.video_id = v.id
        LEFT JOIN users u ON p.user_id = u.id
        WHERE p.id = ?
    """, (prediction_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_predictions(user_id, limit=50):
    """Fetches prediction history for a specific user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            p.*, 
            v.filename, v.original_name, v.filepath, v.source_type, v.source_url
        FROM predictions p
        JOIN videos v ON p.video_id = v.id
        WHERE p.user_id = ?
        ORDER BY p.created_at DESC
        LIMIT ?
    """, (user_id, limit))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_all_predictions(limit=100):
    """Fetches all system predictions for admin view."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            p.*, 
            v.filename, v.original_name, v.filepath, v.source_type, v.source_url,
            u.username, u.email
        FROM predictions p
        JOIN videos v ON p.video_id = v.id
        LEFT JOIN users u ON p.user_id = u.id
        ORDER BY p.created_at DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


# ---------------- TRAINING HISTORY FUNCTIONS ---------------- #

def log_training_run(admin_id, epochs, batch_size, train_loss, train_accuracy, val_loss, val_accuracy, checkpoint_path, real_count, fake_count, status="completed"):
    """Logs a completed training session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO training_history (admin_id, epochs, batch_size, train_loss, train_accuracy, val_loss, val_accuracy, checkpoint_path, dataset_real_count, dataset_fake_count, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (admin_id, epochs, batch_size, train_loss, train_accuracy, val_loss, val_accuracy, checkpoint_path, real_count, fake_count, status))
    history_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return history_id


def get_training_history(limit=20):
    """Fetches recent training runs."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT th.*, u.username as admin_username
        FROM training_history th
        LEFT JOIN users u ON th.admin_id = u.id
        ORDER BY th.created_at DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_latest_training_stats():
    """Fetches the latest completed training record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM training_history
        WHERE status = 'completed'
        ORDER BY created_at DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


# ---------------- DASHBOARD AGGREGATE STATS ---------------- #

def get_system_statistics():
    """Fetches high-level metrics for admin dashboard."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'user'")
    total_users = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM videos")
    total_videos = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction_label = 'REAL'")
    total_real = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction_label = 'FAKE'")
    total_fake = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]

    conn.close()
    return {
        "total_users": total_users,
        "total_videos": total_videos,
        "total_predictions": total_predictions,
        "total_real": total_real,
        "total_fake": total_fake
    }


def get_user_statistics(user_id):
    """Fetches statistics for a specific user dashboard."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM videos WHERE user_id = ?", (user_id,))
    total_videos = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = ?", (user_id,))
    total_predictions = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = ? AND prediction_label = 'REAL'", (user_id,))
    total_real = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = ? AND prediction_label = 'FAKE'", (user_id,))
    total_fake = cursor.fetchone()[0]

    conn.close()
    return {
        "total_videos": total_videos,
        "total_predictions": total_predictions,
        "total_real": total_real,
        "total_fake": total_fake
    }


# Auto-initialize database on import if not already created
init_db()

if __name__ == "__main__":
    init_db()
    print("[Database] Initialized successfully.")
    stats = get_system_statistics()
    print(f"[Database] Initial Stats: {stats}")
