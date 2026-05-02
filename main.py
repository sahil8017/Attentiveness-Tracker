"""
Attentiveness Tracker — FastAPI Application
=============================================
Real-time attentiveness detection using Roboflow AI (RF-DETR Nano)
with session management, temporal smoothing, interactive analytics,
and JWT-based multi-user authentication.

Uses direct Roboflow Inference HTTP API (httpx) for fast async predictions.
"""

import cv2
import csv
import io
import base64
import uuid
import logging
import numpy as np
import httpx
from datetime import datetime
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import func

from config import Config
from db import init_db, get_db
from models import Session, Detection, User
from auth import get_current_user, get_optional_user
from routes.auth_routes import router as auth_router

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
http_client: httpx.AsyncClient = None

# === IN-MEMORY STATE ===
smoothing_buffers = {}
frame_counters = {}
alert_counters = {}
last_known_states = {}


# === PYDANTIC MODELS ===

class PredictRequest(BaseModel):
    image: str
    session_id: str = "default"

class CreateSessionRequest(BaseModel):
    device_id: Optional[str] = None

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
    global http_client

    logger.info("=" * 50)
    logger.info("Attentiveness Tracker v3.0 — Starting")
    logger.info("=" * 50)

    # Validate config
    Config.validate()

    # Initialize SQLAlchemy database
    init_db(Config.DATABASE_URL)
    logger.info(f"Database initialized: {Config.DATABASE_URL.split('@')[-1] if '@' in Config.DATABASE_URL else Config.DATABASE_URL}")

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
    description="AI-powered real-time focus monitoring with multi-user auth",
    version="3.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=Config.BASE_DIR / "static"), name="static")

# Templates
templates = Jinja2Templates(directory=Config.BASE_DIR / "templates")

# Include auth router
app.include_router(auth_router)


# === PAGE ROUTES (public) ===

@app.get("/")
async def index(request: Request):
    """Homepage."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/detection")
async def detection_page(request: Request):
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
async def dashboard_page(request: Request):
    """Analytics dashboard page."""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/login")
async def login_page(request: Request):
    """Login page."""
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register")
async def register_page(request: Request):
    """Registration page."""
    return templates.TemplateResponse("register.html", {"request": request})


# === API ENDPOINTS ===

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "3.0.0", "model": Config.ROBOFLOW_PROJECT}


# --- Session Management (protected) ---

@app.post("/api/sessions")
def create_session(
    data: CreateSessionRequest = CreateSessionRequest(),
    current_user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """Create a new detection session (requires auth)."""
    try:
        session_id = f"session-{uuid.uuid4().hex[:12]}"
        session = Session(
            id=session_id,
            user_id=current_user.id,
            device_id=data.device_id,
            start_time=datetime.utcnow(),
        )
        db.add(session)
        db.commit()
        logger.info(f"Session created: {session_id} (user={current_user.id}, device={data.device_id})")
        return {"success": True, "session_id": session_id}
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create session: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/sessions/{session_id}/end")
def end_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """End a detection session and compute attention score (requires auth)."""
    try:
        session = db.query(Session).filter(
            Session.id == session_id,
            Session.user_id == current_user.id,
        ).first()
        if not session:
            return JSONResponse({"success": False, "error": "Session not found"}, status_code=404)

        # Calculate attention score from detections
        total = db.query(func.count(Detection.id)).filter(
            Detection.session_id == session_id
        ).scalar() or 0

        attentive = db.query(func.count(Detection.id)).filter(
            Detection.session_id == session_id,
            Detection.class_name == "awake",
        ).scalar() or 0

        score = round((attentive / total * 100), 1) if total > 0 else 0.0

        session.end_time = datetime.utcnow()
        session.total_frames = total
        session.attention_score = score
        db.commit()

        # Cleanup in-memory state
        smoothing_buffers.pop(session_id, None)
        frame_counters.pop(session_id, None)
        alert_counters.pop(session_id, None)
        last_known_states.pop(session_id, None)

        logger.info(f"Session ended: {session_id} (score={score}%)")
        return {"success": True, "attention_score": score}
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to end session: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/sessions")
def list_sessions(
    limit: int = 20,
    device_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """List recent sessions for the current user (requires auth)."""
    try:
        query = db.query(Session).filter(Session.user_id == current_user.id)
        if device_id:
            query = query.filter(Session.device_id == device_id)
        sessions = query.order_by(Session.start_time.desc()).limit(limit).all()
        return {"success": True, "sessions": [s.to_dict() for s in sessions]}
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.delete("/api/sessions/{session_id}")
def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """Delete a specific session (requires auth, must own session)."""
    try:
        session = db.query(Session).filter(
            Session.id == session_id,
            Session.user_id == current_user.id,
        ).first()
        if not session:
            return JSONResponse({"success": False, "error": "Session not found"}, status_code=404)

        db.delete(session)  # cascade will delete detections
        db.commit()

        smoothing_buffers.pop(session_id, None)
        frame_counters.pop(session_id, None)
        alert_counters.pop(session_id, None)
        last_known_states.pop(session_id, None)

        logger.info(f"Session deleted: {session_id}")
        return {"success": True, "message": f"Session {session_id} deleted"}
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete session: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.delete("/api/sessions")
def clear_all_sessions(
    current_user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """Delete ALL sessions for the current user."""
    try:
        user_sessions = db.query(Session).filter(Session.user_id == current_user.id).all()
        for s in user_sessions:
            smoothing_buffers.pop(s.id, None)
            frame_counters.pop(s.id, None)
            alert_counters.pop(s.id, None)
            last_known_states.pop(s.id, None)
            db.delete(s)
        db.commit()
        logger.info(f"All sessions cleared for user {current_user.id}")
        return {"success": True, "message": "All sessions and data cleared"}
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to clear sessions: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.delete("/api/detections")
def clear_all_detections(
    current_user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """Delete ALL detection records for the current user (reset dashboard)."""
    try:
        # Get all session IDs for this user
        session_ids = [s.id for s in db.query(Session.id).filter(
            Session.user_id == current_user.id
        ).all()]

        if session_ids:
            db.query(Detection).filter(Detection.session_id.in_(session_ids)).delete(synchronize_session=False)
            db.query(Session).filter(Session.id.in_(session_ids)).update(
                {"total_frames": 0, "attention_score": 0.0},
                synchronize_session=False,
            )
            db.commit()

        logger.info(f"All detection data cleared for user {current_user.id}")
        return {"success": True, "message": "All detection data cleared"}
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to clear detections: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# --- Prediction (protected) ---

@app.post("/api/predict")
@app.post("/predict")  # Fallback for cached clients
async def predict(
    data: PredictRequest,
    current_user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """
    Receives a base64 image and returns Roboflow predictions
    with temporal smoothing and quality checks (requires auth).
    """
    try:
        session_id = data.session_id

        # Verify session belongs to user
        session = db.query(Session).filter(
            Session.id == session_id,
            Session.user_id == current_user.id,
        ).first()
        if not session:
            return JSONResponse({"error": "Session not found or unauthorized"}, status_code=403)

        # Initialize frame counter for session
        if session_id not in frame_counters:
            frame_counters[session_id] = 0

        # Decode base64 image
        img_data = data.image.split(",")[1] if "," in data.image else data.image
        # Add padding if needed
        missing_padding = len(img_data) % 4
        if missing_padding:
            img_data += "=" * (4 - missing_padding)
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse({"error": "Invalid image data"}, status_code=400)

        # Frame quality check
        is_clear, blur_score = check_frame_quality(frame)
        if not is_clear:
            last_state = last_known_states.get(session_id)
            if last_state:
                return {
                    "success": True, "skipped": False, "blurry": True,
                    "blur_score": blur_score,
                    "predictions": last_state["predictions"],
                    "frame_id": last_state["frame_id"],
                    "trigger_alert": False,
                }
            else:
                return {
                    "success": True, "skipped": True,
                    "reason": "Frame too blurry (waiting for clear frame)",
                    "blur_score": blur_score, "predictions": [],
                }

        # Increment frame counter
        frame_counters[session_id] += 1
        frame_id = frame_counters[session_id]

        # Send to Roboflow Inference API
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

            smoothed = get_smoothed_class(session_id, raw_class)

            if should_alert(session_id, smoothed):
                trigger_alert = True

            # Log to database
            detection = Detection(
                session_id=session_id,
                timestamp=datetime.utcnow(),
                class_name=raw_class,
                confidence=confidence,
                frame_id=frame_id,
                smoothed_class=smoothed,
            )
            db.add(detection)

            processed_predictions.append({
                "x": pred["x"], "y": pred["y"],
                "width": pred["width"], "height": pred["height"],
                "class": raw_class, "smoothed_class": smoothed,
                "confidence": confidence,
            })

        db.commit()

        # Update Session-level stats in real-time
        try:
            # Count total detections for this session
            total_dets = db.query(Detection.id).filter(Detection.session_id == session_id).count()
            # Count 'engaged' detections (case-insensitive)
            engaged_dets = db.query(Detection.id).filter(
                Detection.session_id == session_id,
                Detection.class_name == "engaged"
            ).count()

            # Attention score = (engaged detections / total detections) * 100
            new_score = round((engaged_dets / total_dets * 100), 1) if total_dets > 0 else 0

            # Update session record
            session.total_frames = frame_id  # Tracking frames processed
            session.attention_score = new_score
            db.commit()
            logger.info(f"Session {session_id} stats updated: frames={frame_id}, score={new_score}%")
        except Exception as e:
            logger.error(f"Error updating session stats: {e}")
            db.rollback()

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
            "trigger_alert": trigger_alert,
        }

    except httpx.TimeoutException:
        logger.warning(f"Roboflow API timeout for session {data.session_id}")
        last_state = last_known_states.get(data.session_id)
        if last_state:
            return {
                "success": True, "skipped": False, "timeout": True,
                "predictions": last_state["predictions"],
                "frame_id": last_state["frame_id"],
                "trigger_alert": False,
            }
        return {"success": True, "skipped": True, "reason": "API timeout", "predictions": []}

    except httpx.HTTPStatusError as e:
        logger.error(f"Roboflow API error: {e.response.status_code} - {e.response.text}")
        return JSONResponse({"error": f"Roboflow API error: {e.response.status_code}"}, status_code=502)

    except Exception as e:
        db.rollback()
        logger.error(f"Prediction error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# --- Analytics (protected) ---

@app.get("/api/get_stats")
@app.get("/get_stats")  # Fallback for cached clients
def get_stats(
    session_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """Returns aggregate stats for the current user."""
    try:
        # Get user's session IDs
        user_session_ids = [s.id for s in db.query(Session.id).filter(
            Session.user_id == current_user.id
        ).all()]

        if session_id:
            if session_id not in user_session_ids:
                return {"total_frames": 0, "avg_confidence": 0, "classes": {}}
            user_session_ids = [session_id]

        if not user_session_ids:
            return {"total_frames": 0, "avg_confidence": 0, "classes": {}}

        q = db.query(
            func.count(Detection.id).label("total_frames"),
            func.avg(Detection.confidence).label("avg_confidence"),
        ).filter(Detection.session_id.in_(user_session_ids)).first()

        class_rows = db.query(
            Detection.class_name,
            func.count(Detection.id).label("count"),
        ).filter(
            Detection.session_id.in_(user_session_ids)
        ).group_by(Detection.class_name).order_by(func.count(Detection.id).desc()).all()

        return {
            "total_frames": q.total_frames or 0,
            "avg_confidence": round(float(q.avg_confidence or 0), 3),
            "classes": {r.class_name: r.count for r in class_rows},
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/chart_data")
@app.get("/chart_data")  # Fallback for cached clients
def chart_data(
    session_id: Optional[str] = None,
    limit: int = 500,
    current_user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """Returns time-series data for Chart.js (user-scoped)."""
    try:
        user_session_ids = [s.id for s in db.query(Session.id).filter(
            Session.user_id == current_user.id
        ).all()]

        if session_id:
            if session_id not in user_session_ids:
                return {"success": True, "data": {"labels": [], "confidence": [], "classes": [], "frame_ids": []}}
            user_session_ids = [session_id]

        if not user_session_ids:
            return {"success": True, "data": {"labels": [], "confidence": [], "classes": [], "frame_ids": []}}

        rows = db.query(Detection).filter(
            Detection.session_id.in_(user_session_ids)
        ).order_by(Detection.timestamp.desc()).limit(limit).all()

        data = list(reversed(rows))

        return {
            "success": True,
            "data": {
                "labels": [d.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ") if d.timestamp else "" for d in data],
                "confidence": [d.confidence for d in data],
                "classes": [d.smoothed_class or d.class_name for d in data],
                "frame_ids": [d.frame_id for d in data],
            },
        }
    except Exception as e:
        logger.error(f"Error getting chart data: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/session_scores")
def session_scores(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """Attention scores across sessions for trend chart (user-scoped)."""
    try:
        sessions = db.query(Session).filter(
            Session.user_id == current_user.id,
            Session.end_time.is_not(None),
        ).order_by(Session.start_time.desc()).limit(limit).all()

        scores = [s.to_dict() for s in reversed(sessions)]
        return {"success": True, "scores": scores}
    except Exception as e:
        logger.error(f"Error getting session scores: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/export")
def export_data(
    session_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """Export detections as CSV (user-scoped)."""
    try:
        user_session_ids = [s.id for s in db.query(Session.id).filter(
            Session.user_id == current_user.id
        ).all()]

        if session_id:
            if session_id not in user_session_ids:
                return JSONResponse({"error": "Session not found"}, status_code=404)
            user_session_ids = [session_id]

        detections = db.query(Detection).filter(
            Detection.session_id.in_(user_session_ids)
        ).order_by(Detection.timestamp.asc()).all()

        if not detections:
            return JSONResponse({"error": "No data to export"}, status_code=404)

        output = io.StringIO()
        fieldnames = ["timestamp", "class", "confidence", "frame_id", "smoothed_class", "session_id"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for d in detections:
            writer.writerow(d.to_dict())

        csv_content = output.getvalue()
        filename = f"attentiveness_export_{session_id or 'all'}.csv"
        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        logger.error(f"Export error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# === ERROR HANDLERS ===

@app.exception_handler(404)
async def not_found(request: Request, exc):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": "Not found"}, status_code=404)
    return templates.TemplateResponse("index.html", {"request": request}, status_code=404)

@app.exception_handler(500)
async def server_error(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": "Internal server error"}, status_code=500)
    return templates.TemplateResponse("index.html", {"request": request}, status_code=500)


# === ENTRY POINT ===
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Attentiveness Tracker (FastAPI v3.0)...")
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=Config.DEBUG)
