"""
Attentiveness Tracker — FastAPI Application
=============================================
Real-time attentiveness detection using Roboflow AI (RF-DETR Nano)
with session management, temporal smoothing, and interactive analytics.

Uses direct Roboflow Inference HTTP API (httpx) for fast async predictions.
"""

import cv2
import base64
import uuid
import logging
import numpy as np
import httpx
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import Config
from database import Database

# === LOGGING ===
logging.basicConfig(
    level=logging.DEBUG if Config.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Config.LOGS_DIR / "app.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

# === GLOBALS (initialized at startup) ===
db: Database = None
http_client: httpx.AsyncClient = None

# === IN-MEMORY STATE ===
smoothing_buffers = {}
frame_counters = {}
alert_counters = {}
last_known_states = {}  # Track last known state per session for blur fallback


# === PYDANTIC MODELS ===

class PredictRequest(BaseModel):
    image: str
    session_id: str = "default"

class SuccessResponse(BaseModel):
    success: bool = True
    message: str = ""


# === HELPER FUNCTIONS ===

def get_smoothed_class(session_id, raw_class):
    """Apply temporal smoothing using majority vote over recent predictions."""
    if session_id not in smoothing_buffers:
        smoothing_buffers[session_id] = deque(maxlen=Config.SMOOTHING_WINDOW)
    smoothing_buffers[session_id].append(raw_class)
    counts = {}
    for c in smoothing_buffers[session_id]:
        counts[c] = counts.get(c, 0) + 1
    return max(counts, key=counts.get)


def check_frame_quality(frame):
    """Check if frame is too blurry using Laplacian variance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance >= Config.BLUR_THRESHOLD, round(variance, 1)


def should_alert(session_id, smoothed_class):
    """Only alert after consecutive inattentive frames exceed threshold."""
    if smoothed_class in ("sleepy", "bored"):
        alert_counters[session_id] = alert_counters.get(session_id, 0) + 1
        if alert_counters[session_id] >= Config.ALERT_CONSECUTIVE_FRAMES:
            return True
    else:
        alert_counters[session_id] = 0
    return False


# === LIFESPAN ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global db, http_client

    logger.info("=" * 50)
    logger.info("Attentiveness Tracker v2.1 — Starting")
    logger.info("=" * 50)

    # Validate config
    Config.validate()

    # Initialize database
    db = Database(Config.DATABASE_PATH)
    logger.info(f"Database initialized: {Config.DATABASE_PATH}")

    # Initialize async HTTP client with connection pooling
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(Config.ROBOFLOW_TIMEOUT, connect=5.0),
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    logger.info("Async HTTP client initialized (httpx)")

    # Build Roboflow inference URL
    inference_url = (
        f"{Config.ROBOFLOW_API_URL}/{Config.ROBOFLOW_PROJECT}/{Config.ROBOFLOW_VERSION}"
        f"?api_key={Config.ROBOFLOW_API_KEY}"
        f"&confidence={Config.CONFIDENCE_THRESHOLD}"
        f"&overlap={Config.OVERLAP_THRESHOLD}"
    )
    app.state.inference_url = inference_url
    logger.info(f"Roboflow Inference API configured: {Config.ROBOFLOW_PROJECT} v{Config.ROBOFLOW_VERSION}")

    yield

    # Shutdown
    await http_client.aclose()
    logger.info("HTTP client closed. Goodbye!")


# === APP ===

app = FastAPI(
    title="Attentiveness Tracker",
    description="AI-powered real-time focus monitoring",
    version="2.1.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory=Config.BASE_DIR / "static"), name="static")

# Templates
templates = Jinja2Templates(directory=Config.BASE_DIR / "templates")


# === PAGE ROUTES ===

@app.get("/")
async def index(request: Request):
    """Homepage."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/detection")
async def detection(request: Request):
    """Live detection page."""
    return templates.TemplateResponse("detection.html", {
        "request": request,
        "config": {
            "blur_threshold": Config.BLUR_THRESHOLD,
            "smoothing_window": Config.SMOOTHING_WINDOW,
            "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
        }
    })

@app.get("/dashboard")
async def dashboard(request: Request):
    """Analytics dashboard page."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


# === API ENDPOINTS ===

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.1.0", "model": Config.ROBOFLOW_PROJECT}

@app.post("/api/sessions")
async def create_session():
    """Create a new detection session."""
    try:
        session_id = f"session-{uuid.uuid4().hex[:12]}"
        db.create_session(session_id)
        logger.info(f"Session created: {session_id}")
        return {"success": True, "session_id": session_id}
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/api/sessions/{session_id}/end")
async def end_session(session_id: str):
    """End a detection session."""
    try:
        result = db.end_session(session_id)
        # Cleanup in-memory state
        smoothing_buffers.pop(session_id, None)
        frame_counters.pop(session_id, None)
        alert_counters.pop(session_id, None)
        last_known_states.pop(session_id, None)
        logger.info(f"Session ended: {session_id}")
        return {"success": True, "attention_score": result.get("attention_score")}
    except Exception as e:
        logger.error(f"Failed to end session: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/sessions")
async def list_sessions(limit: int = 20):
    """List recent detection sessions."""
    try:
        sessions = db.get_sessions(limit)
        return {"success": True, "sessions": sessions}
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session and its detections."""
    try:
        db.delete_session(session_id)
        smoothing_buffers.pop(session_id, None)
        frame_counters.pop(session_id, None)
        alert_counters.pop(session_id, None)
        last_known_states.pop(session_id, None)
        logger.info(f"Session deleted: {session_id}")
        return {"success": True, "message": f"Session {session_id} deleted"}
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.delete("/api/sessions")
async def clear_all_sessions():
    """Delete ALL sessions and detection data."""
    try:
        db.clear_all_sessions()
        smoothing_buffers.clear()
        frame_counters.clear()
        alert_counters.clear()
        last_known_states.clear()
        logger.info("All sessions cleared")
        return {"success": True, "message": "All sessions and data cleared"}
    except Exception as e:
        logger.error(f"Failed to clear sessions: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.delete("/api/detections")
async def clear_all_detections():
    """Delete ALL detection records (reset dashboard analytics)."""
    try:
        db.clear_all_detections()
        logger.info("All detection data cleared")
        return {"success": True, "message": "All detection data cleared"}
    except Exception as e:
        logger.error(f"Failed to clear detections: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/predict")
async def predict(data: PredictRequest):
    """
    Receives a base64 image and returns Roboflow predictions
    with temporal smoothing and quality checks.

    Uses direct HTTP API call to Roboflow Inference — no temp files.
    """
    try:
        session_id = data.session_id

        # Initialize frame counter for session
        if session_id not in frame_counters:
            frame_counters[session_id] = 0

        # Decode base64 image
        img_data = data.image.split(",")[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse({"error": "Invalid image data"}, status_code=400)

        # Frame quality check
        is_clear, blur_score = check_frame_quality(frame)
        if not is_clear:
            # Graceful degradation: return last known state instead of skipping
            last_state = last_known_states.get(session_id)
            if last_state:
                return {
                    "success": True,
                    "skipped": False,
                    "blurry": True,
                    "blur_score": blur_score,
                    "predictions": last_state["predictions"],
                    "frame_id": last_state["frame_id"],
                    "trigger_alert": False,
                }
            else:
                # No previous state — still return skipped but with lower priority
                return {
                    "success": True,
                    "skipped": True,
                    "reason": "Frame too blurry (waiting for clear frame)",
                    "blur_score": blur_score,
                    "predictions": []
                }

        # Increment frame counter
        frame_counters[session_id] += 1
        frame_id = frame_counters[session_id]

        # Send base64 directly to Roboflow Inference API (no temp file!)
        response = await http_client.post(
            app.state.inference_url,
            data=img_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        prediction = response.json()

        logger.debug(f"Raw prediction response: {prediction}")
        logger.info(f"Frame {frame_id}: {len(prediction.get('predictions', []))} detections found")

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

        # Store last known state for blur fallback
        last_known_states[session_id] = {
            "predictions": processed_predictions,
            "frame_id": frame_id,
        }

        return {
            "success": True,
            "predictions": processed_predictions,
            "frame_id": frame_id,
            "blur_score": blur_score,
            "trigger_alert": trigger_alert
        }

    except httpx.TimeoutException:
        logger.warning(f"Roboflow API timeout for session {data.session_id}")
        # Return last known state on timeout
        last_state = last_known_states.get(data.session_id)
        if last_state:
            return {
                "success": True,
                "skipped": False,
                "timeout": True,
                "predictions": last_state["predictions"],
                "frame_id": last_state["frame_id"],
                "trigger_alert": False,
            }
        return {"success": True, "skipped": True, "reason": "API timeout", "predictions": []}

    except httpx.HTTPStatusError as e:
        logger.error(f"Roboflow API error: {e.response.status_code} - {e.response.text}")
        return JSONResponse({"error": f"Roboflow API error: {e.response.status_code}"}, status_code=502)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/get_stats")
async def get_stats(session_id: str = None):
    """Returns statistics, optionally filtered by session."""
    try:
        stats = db.get_stats(session_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/chart_data")
async def chart_data(session_id: str = None, limit: int = 500):
    """Returns time-series data formatted for Chart.js."""
    try:
        data = db.get_chart_data(session_id, limit)
        return {"success": True, "data": data}
    except Exception as e:
        logger.error(f"Error getting chart data: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/session_scores")
async def session_scores(limit: int = 20):
    """Returns attention scores across sessions for trend chart."""
    try:
        scores = db.get_session_scores(limit)
        return {"success": True, "scores": scores}
    except Exception as e:
        logger.error(f"Error getting session scores: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/export")
async def export_data(session_id: str = None):
    """Export detections as CSV."""
    import csv
    import io

    try:
        detections = db.get_detections_for_export(session_id)

        if not detections:
            return JSONResponse({"error": "No data to export"}, status_code=404)

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=detections[0].keys())
        writer.writeheader()
        writer.writerows(detections)

        csv_content = output.getvalue()

        filename = f"attentiveness_export_{session_id or 'all'}.csv"
        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"Export error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# === ERROR HANDLERS ===

@app.exception_handler(404)
async def not_found(request: Request, exc):
    """Handle 404 errors."""
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": "Not found"}, status_code=404)
    return templates.TemplateResponse("index.html", {"request": request}, status_code=404)

@app.exception_handler(500)
async def server_error(request: Request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": "Internal server error"}, status_code=500)
    return templates.TemplateResponse("index.html", {"request": request}, status_code=500)


# === ENTRY POINT ===
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Attentiveness Tracker (FastAPI)...")
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=Config.DEBUG)
