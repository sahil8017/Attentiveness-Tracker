"""
Attentiveness Tracker — Flask Application
==========================================
Real-time attentiveness detection using Roboflow AI with session management,
temporal smoothing, and interactive analytics.
"""

import cv2
import base64
import uuid
import logging
import json
import csv
import io
import numpy as np
from collections import deque
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from roboflow import Roboflow

from config import Config
from database import Database

# === LOGGING ===
Config.validate()
Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(Config.LOGS_DIR / "app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === INITIALIZATION ===
app = Flask(__name__)
app.secret_key = Config.SECRET_KEY

# Database
db = Database(Config.DATABASE_PATH)

# Migrate existing CSV if present
csv_path = Config.BASE_DIR / "attentiveness_log.csv"
if csv_path.exists():
    count = db.migrate_from_csv(csv_path)
    if count > 0:
        logger.info(f"Migrated {count} records from CSV to database")

# Roboflow model
logger.info("Loading Roboflow model...")
rf = Roboflow(api_key=Config.ROBOFLOW_API_KEY)
project = rf.workspace().project(Config.ROBOFLOW_PROJECT)
model = project.version(Config.ROBOFLOW_VERSION).model
logger.info("Roboflow model loaded successfully")

# === IN-MEMORY STATE ===
# Per-session smoothing buffers: { session_id: deque([class1, class2, ...]) }
smoothing_buffers = {}
frame_counters = {}
alert_counters = {}  # Consecutive inattentive frame counts per session


def get_smoothed_class(session_id, raw_class):
    """Apply temporal smoothing using majority vote over recent predictions."""
    if session_id not in smoothing_buffers:
        smoothing_buffers[session_id] = deque(maxlen=Config.SMOOTHING_WINDOW)

    smoothing_buffers[session_id].append(raw_class)
    buffer = smoothing_buffers[session_id]

    # Majority vote
    from collections import Counter
    counts = Counter(buffer)
    return counts.most_common(1)[0][0]


def check_frame_quality(frame):
    """Check if frame is too blurry using Laplacian variance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance >= Config.BLUR_THRESHOLD, round(variance, 1)


def should_alert(session_id, smoothed_class):
    """Only alert after consecutive inattentive frames exceed threshold."""
    alert_classes = ["sleepy", "bored"]

    if session_id not in alert_counters:
        alert_counters[session_id] = 0

    if smoothed_class in alert_classes:
        alert_counters[session_id] += 1
    else:
        alert_counters[session_id] = 0

    return alert_counters[session_id] >= Config.ALERT_CONSECUTIVE_FRAMES


# === PAGE ROUTES ===

@app.route("/")
def index():
    """Homepage."""
    return render_template("index.html")


@app.route("/detection")
def detection():
    """Live detection page."""
    return render_template("detection.html", config={
        "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
        "smoothing_window": Config.SMOOTHING_WINDOW,
        "alert_consecutive_frames": Config.ALERT_CONSECUTIVE_FRAMES
    })


@app.route("/dashboard")
def dashboard():
    """Analytics dashboard page."""
    return render_template("dashboard.html")


# === API ENDPOINTS ===

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


@app.route("/api/sessions", methods=["POST"])
def create_session():
    """Create a new detection session."""
    try:
        session_id = f"session-{uuid.uuid4().hex[:12]}"
        db.create_session(session_id)
        smoothing_buffers[session_id] = deque(maxlen=Config.SMOOTHING_WINDOW)
        frame_counters[session_id] = 0
        alert_counters[session_id] = 0
        logger.info(f"Session created: {session_id}")
        return jsonify({"success": True, "session_id": session_id})
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/sessions/<session_id>/end", methods=["POST"])
def end_session(session_id):
    """End a detection session."""
    try:
        score = db.end_session(session_id)
        # Cleanup in-memory state
        smoothing_buffers.pop(session_id, None)
        frame_counters.pop(session_id, None)
        alert_counters.pop(session_id, None)
        logger.info(f"Session ended: {session_id}, score: {score}%")
        return jsonify({"success": True, "attention_score": score})
    except Exception as e:
        logger.error(f"Error ending session: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    """List recent detection sessions."""
    try:
        limit = request.args.get("limit", 20, type=int)
        sessions = db.get_sessions(limit=limit)
        return jsonify({"success": True, "sessions": sessions})
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives a base64 image and returns Roboflow predictions
    with temporal smoothing and quality checks.
    """
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        session_id = data.get("session_id", "default")

        # Initialize frame counter for session
        if session_id not in frame_counters:
            frame_counters[session_id] = 0

        # Decode base64 image
        img_data = data["image"].split(",")[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Frame quality check
        is_clear, blur_score = check_frame_quality(frame)
        if not is_clear:
            return jsonify({
                "success": True,
                "skipped": True,
                "reason": "Frame too blurry",
                "blur_score": blur_score,
                "predictions": []
            })

        # Increment frame counter
        frame_counters[session_id] += 1
        frame_id = frame_counters[session_id]

        # Save temporary frame for Roboflow
        temp_path = Config.TEMP_DIR / f"frame_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(temp_path), frame)

        # Roboflow prediction
        prediction = model.predict(
            str(temp_path),
            confidence=Config.CONFIDENCE_THRESHOLD,
            overlap=Config.OVERLAP_THRESHOLD
        ).json()

        # Delete temp image
        if temp_path.exists():
            temp_path.unlink()

        # Process predictions with smoothing
        processed_predictions = []
        trigger_alert = False

        for pred in prediction.get("predictions", []):
            raw_class = pred["class"].lower()
            confidence = round(pred["confidence"], 3)

            # Temporal smoothing
            smoothed = get_smoothed_class(session_id, raw_class)

            # Check alert condition
            if should_alert(session_id, smoothed):
                trigger_alert = True

            # Log to database
            db.log_detection(session_id, raw_class, confidence, frame_id, smoothed)

            processed_predictions.append({
                "x": pred["x"],
                "y": pred["y"],
                "width": pred["width"],
                "height": pred["height"],
                "class": raw_class,
                "smoothed_class": smoothed,
                "confidence": confidence
            })

        return jsonify({
            "success": True,
            "predictions": processed_predictions,
            "frame_id": frame_id,
            "blur_score": blur_score,
            "trigger_alert": trigger_alert
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/get_stats", methods=["GET"])
def get_stats():
    """Returns statistics, optionally filtered by session."""
    try:
        session_id = request.args.get("session_id")
        stats = db.get_stats(session_id=session_id)
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chart_data", methods=["GET"])
def chart_data():
    """Returns time-series data formatted for Chart.js."""
    try:
        session_id = request.args.get("session_id")
        limit = request.args.get("limit", 500, type=int)
        data = db.get_chart_data(session_id=session_id, limit=limit)
        return jsonify({"success": True, "data": data})
    except Exception as e:
        logger.error(f"Chart data error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/session_scores", methods=["GET"])
def session_scores():
    """Returns attention scores across sessions for trend chart."""
    try:
        limit = request.args.get("limit", 20, type=int)
        scores = db.get_session_scores(limit=limit)
        return jsonify({"success": True, "scores": scores})
    except Exception as e:
        logger.error(f"Session scores error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/export", methods=["GET"])
def export_data():
    """Export detections as CSV."""
    try:
        session_id = request.args.get("session_id")
        detections = db.get_detections_for_export(session_id=session_id)

        if not detections:
            return jsonify({"error": "No data to export"}), 404

        # Build CSV in memory
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            "timestamp", "session_id", "class", "confidence",
            "frame_id", "smoothed_class"
        ])
        writer.writeheader()
        writer.writerows(detections)

        csv_content = output.getvalue()
        output.close()

        filename = f"attentiveness_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        logger.error(f"Export error: {e}")
        return jsonify({"error": str(e)}), 500


# === ERROR HANDLERS ===

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    if request.path.startswith("/api/"):
        return jsonify({"error": "Endpoint not found"}), 404
    return render_template("index.html"), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Server error: {e}")
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server error"}), 500
    return render_template("index.html"), 500


# === ENTRY POINT ===
if __name__ == "__main__":
    logger.info("Starting Attentiveness Tracker...")
    app.run(host="0.0.0.0", port=5000, debug=Config.DEBUG)
