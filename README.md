---
title: Attentiveness Tracker
emoji: "📊"
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
---

<div align="center">

# 🎯 Attentiveness Tracker

<p align="center">
  <b>AI-Powered Real-Time Focus Monitoring — Know When You Drift. Stay Sharp.</b>
</p>

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Roboflow](https://img.shields.io/badge/Roboflow-RF--DETR_Nano-6200EE?style=for-the-badge)](https://roboflow.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Neon](https://img.shields.io/badge/Neon-Serverless_DB-00E599?style=for-the-badge)](https://neon.tech/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-orange?style=for-the-badge)](https://huggingface.co/spaces/sahil8017/Attentiveness_Tracker)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

**[🌐 Live Demo](https://sahil8017-attentiveness-tracker.hf.space/)** &nbsp;|&nbsp; **[📖 Swagger API Docs](https://sahil8017-attentiveness-tracker.hf.space/docs)** &nbsp;|&nbsp; **[⭐ Star this Repo](https://github.com/sahil8017/Attentiveness-Tracker)**

</div>

---

## 📖 Table of Contents

1. [What Is This?](#-what-is-this)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Technology Stack](#-technology-stack)
5. [Detection Pipeline — How It Works](#-detection-pipeline--how-it-works)
6. [REST API Reference](#-rest-api-reference)
7. [Configuration Reference](#-configuration-reference)
8. [Database Schema](#-database-schema)
9. [Authentication Flow](#-authentication-flow)
10. [Free Cloud Deployment Guide](#-free-cloud-deployment-guide)
11. [Local Development with Docker Compose](#-local-development-with-docker-compose)
12. [Project Structure](#-project-structure)
13. [Acknowledgements](#-acknowledgements)

---

## 💡 What Is This?

**Attentiveness Tracker** is a production-grade, full-stack web application that uses a state-of-the-art computer vision model to monitor your attentiveness level in real time through your webcam. It requires no app installation — it runs entirely in the browser.

Every second, it captures a frame from your webcam, sends it to an AI model (RF-DETR Nano via Roboflow), and classifies your state as:

| State | Meaning |
|-------|---------|
| 🟢 **Awake** | You are alert, engaged, and focused |
| 🟡 **Bored** | Your engagement level is dropping |
| 🔴 **Sleepy** | You are drowsy or significantly inattentive |

When sustained inattention is detected, a smart audio alert fires — **not on a single bad frame, but only after multiple consecutive inattentive predictions**, preventing false alarms entirely.

**Built for:**
- 📚 Students in long study or exam sessions
- 💼 Remote workers in deep focus or video meetings
- 🧑‍🏫 Educators monitoring classroom engagement
- 🔬 Researchers studying cognitive attention patterns

The entire production stack costs **$0/month**, deployed on Hugging Face Spaces (16 GB RAM Docker container) with a Neon.tech Serverless PostgreSQL database.

---

## ✨ Key Features

| Feature | Details |
|---------|---------|
| ⚡ **Real-Time AI Detection** | Sends webcam frames every ~100ms to the Roboflow RF-DETR Nano model via async httpx. Processes at 15–30 FPS. |
| 🛡️ **Temporal Smoothing** | A majority-vote algorithm over a rolling N-frame buffer prevents rapid state flickering. Only stable, sustained predictions change the displayed state. |
| 🔍 **Frame Quality Gating** | OpenCV computes the Laplacian variance of every frame. Blurry frames are rejected before reaching the model, saving API calls and preventing noise. |
| 🔔 **Smart Audio Alerts** | Alerts only fire after `ALERT_CONSECUTIVE_FRAMES` consecutive inattentive predictions. A single bad frame never triggers a false alarm. |
| 📊 **Interactive Analytics** | Live Chart.js graphs: confidence trend lines, class distribution pie charts, per-session attention scores, and historical session comparison charts. |
| 🔐 **JWT Authentication** | Full multi-user system. Passwords are bcrypt-hashed. Every session and detection is scoped to the authenticated user. JWT tokens expire after a configurable duration. |
| 📱 **Responsive UI** | Glassmorphism-inspired design with automatic Dark/Light mode. Fully responsive from mobile to ultra-wide desktop. |
| ☁️ **100% Free Cloud Stack** | Hugging Face Spaces for compute (2 vCPU / 16 GB RAM). Neon.tech for serverless PostgreSQL. Zero monthly cost. |
| 🐳 **Docker-First** | Production `Dockerfile` + `docker-compose.yml` included. One command spins up the app, database, and Nginx reverse proxy locally. |
| 📁 **CSV Data Export** | Export any session's raw detection data (timestamp, class, confidence, frame_id) as a downloadable CSV for offline analysis. |
| 🔄 **Session Management** | Create, track, end, and delete sessions. Each session stores total frames processed and a computed attention score (% of attentive frames). |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         USER'S WEB BROWSER                               │
│                                                                          │
│  ┌─────────────┐   ┌───────────────────────────────────────────────┐    │
│  │  Webcam     │──►│  JavaScript Frontend                          │    │
│  │  (HTML5)    │   │  - Canvas API (off-screen frame capture)      │    │
│  └─────────────┘   │  - Base64 JPEG compression                   │    │
│                    │  - Chart.js (live analytics)                  │    │
│                    │  - JWT token management (localStorage)        │    │
│                    └───────────────────┬───────────────────────────┘    │
└────────────────────────────────────────┼───────────────────────────────-┘
                                         │ HTTPS POST /api/predict
                                         │ Authorization: Bearer <JWT>
                                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│          🤗 HUGGING FACE SPACES — Docker Container                       │
│                   2 vCPU  /  16 GB RAM  /  FREE                          │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  FastAPI Application  (Uvicorn ASGI, port 7860)                   │  │
│  │                                                                    │  │
│  │  1. JWT Middleware     → Validate Bearer token, extract user_id   │  │
│  │  2. OpenCV Engine      → Laplacian blur detection + rejection     │  │
│  │  3. httpx Client       → Async POST to Roboflow Inference API     │  │
│  │  4. Temporal Smoother  → Majority vote over rolling deque buffer  │  │
│  │  5. Alert Engine       → Count consecutive inattentive frames     │  │
│  │  6. SQLAlchemy ORM     → Write Detection row, update Session row  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬───────────────────────┬───────────────────┘
                               │                       │
                               ▼                       ▼
         ┌───────────────────────────────┐   ┌──────────────────────────┐
         │  🤖 ROBOFLOW INFERENCE API     │   │  🗄️ NEON.TECH            │
         │  RF-DETR Nano Model            │   │  Serverless PostgreSQL 17 │
         │  Project: attention50k v3      │   │  Free Tier (0.5 GB)      │
         │  Returns: bboxes + classes     │   │  Auto-pause when idle    │
         └───────────────────────────────┘   └──────────────────────────┘
```

---

## 🛠️ Technology Stack

### Backend
| Library | Version | Purpose |
|---------|---------|---------|
| FastAPI | 0.115.0 | Async web framework & REST API |
| Uvicorn | 0.30.0 | ASGI server with standard extras |
| httpx | 0.27.0 | Async HTTP client for Roboflow API |
| SQLAlchemy | 2.0.35 | ORM + database session management |
| psycopg2-binary | 2.9.9 | PostgreSQL adapter |
| Jinja2 | 3.1.4 | HTML template rendering |
| opencv-python-headless | 4.10.0 | Frame quality analysis (Laplacian) |
| numpy | 1.26.4 | Image array manipulation |
| Pillow | 10.4.0 | Image processing utilities |
| python-jose | 3.3.0 | JWT token creation & validation |
| bcrypt | 4.2.1 | Password hashing |
| pydantic | 2.9.2 | Request/response schema validation |
| python-dotenv | 1.0.1 | Environment variable loading |

### Frontend
| Technology | Purpose |
|-----------|---------|
| Tailwind CSS | Utility-first styling framework |
| Chart.js | Real-time analytics charts |
| Lucide Icons | Icon library |
| HTML5 Canvas API | Off-screen webcam frame capture |
| HTML5 getUserMedia | Webcam access |
| Web Audio API | Audio alerts |

### Infrastructure (All Free Tier)
| Service | Role | Free Limits |
|---------|------|------------|
| Hugging Face Spaces | App hosting (Docker) | 2 vCPU, 16 GB RAM |
| Neon.tech | Serverless PostgreSQL 17 | 0.5 GB storage, 10 branches |
| Roboflow | AI model inference API | Free tier available |
| GitHub | Source code + version control | Unlimited public repos |

---

## 🔁 Detection Pipeline — How It Works

Each frame goes through 10 deterministic steps from webcam pixel to database record:

```
① CAPTURE
  Browser captures a frame every ~100ms from the webcam using
  an off-screen HTML5 Canvas element. Compresses to Base64 JPEG.

② TRANSMIT
  POST /api/predict with:
  - Body: { image: "data:image/jpeg;base64,...", session_id: "session-xxx" }
  - Header: Authorization: Bearer <JWT_TOKEN>

③ AUTHENTICATE
  FastAPI JWT middleware validates the token.
  Extracts user_id. Returns 401 if invalid or expired.

④ DECODE
  Server base64-decodes the image bytes.
  OpenCV (cv2.imdecode) converts to a NumPy BGR matrix.

⑤ QUALITY GATE
  Converts frame to grayscale.
  Computes cv2.Laplacian variance.
  If variance < BLUR_THRESHOLD → reject frame, return last known state.

⑥ INFERENCE
  Async httpx POST to Roboflow:
  POST https://detect.roboflow.com/{project}/{version}?api_key=...
  Roboflow returns bounding boxes with class names + confidence scores.

⑦ TEMPORAL SMOOTHING
  Each session maintains a deque(maxlen=SMOOTHING_WINDOW).
  The new raw class is appended to the buffer.
  Majority class across the buffer = smoothed_class.
  Prevents flickering between states on ambiguous frames.

⑧ ALERT ENGINE
  If smoothed_class == "sleepy" or "bored":
    alert_counter[session_id] += 1
    if counter >= ALERT_CONSECUTIVE_FRAMES → set trigger_alert=True
  Else:
    alert_counter[session_id] = 0  ← reset on any attentive frame

⑨ PERSIST
  INSERT Detection row: session_id, timestamp, class_name,
  smoothed_class, confidence, frame_id
  UPDATE Session row: total_frames, attention_score (real-time)

⑩ RESPOND
  Returns JSON: predictions[], smoothed_class, blur_score,
  frame_id, trigger_alert to the browser in ~50–200ms.
  Browser draws bounding boxes + updates Chart.js + fires audio if flagged.
```

---

## 📡 REST API Reference

**Base URL:** `https://sahil8017-attentiveness-tracker.hf.space`  
**Interactive Docs:** `/docs` (Swagger UI) | `/redoc` (ReDoc)

All `/api/*` endpoints require `Authorization: Bearer <token>` except auth routes.

### 🔑 Authentication

| Method | Endpoint | Auth | Description |
|--------|----------|:----:|-------------|
| `POST` | `/api/auth/register` | ❌ | Register new user |
| `POST` | `/api/auth/login` | ❌ | Login, receive JWT |
| `GET` | `/api/auth/me` | ✅ | Get current user profile |

**Register:**
```http
POST /api/auth/register
Content-Type: application/json

{ "username": "sahil", "email": "sahil@example.com", "password": "SecurePass123" }
```
```json
{ "success": true, "message": "User registered successfully", "user_id": 1 }
```

**Login:**
```http
POST /api/auth/login
Content-Type: application/json

{ "username": "sahil", "password": "SecurePass123" }
```
```json
{ "access_token": "eyJhbGci...", "token_type": "bearer", "expires_in": 86400 }
```

---

### 🎬 Session Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/sessions` | Create a new tracking session |
| `POST` | `/api/sessions/{id}/end` | End session & compute attention score |
| `GET` | `/api/sessions` | List all user sessions |
| `DELETE` | `/api/sessions/{id}` | Delete one session |
| `DELETE` | `/api/sessions` | Delete all user sessions |

**Create Session:**
```json
// POST /api/sessions
// Body: { "device_id": "optional-fingerprint" }
{ "success": true, "session_id": "session-09d151943dc4" }
```

**End Session:**
```json
// POST /api/sessions/session-09d151943dc4/end
{ "success": true, "attention_score": 87.5 }
```

---

### 🧠 Prediction

**`POST /api/predict`** — Submit a webcam frame for analysis.

```json
// Request
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgAB...",
  "session_id": "session-09d151943dc4"
}

// Response — Successful Detection
{
  "success": true,
  "predictions": [
    {
      "x": 320, "y": 240, "width": 150, "height": 200,
      "class": "awake",
      "smoothed_class": "awake",
      "confidence": 0.942
    }
  ],
  "frame_id": 42,
  "blur_score": 87.3,
  "trigger_alert": false
}

// Response — Blurry Frame Rejected
{
  "success": true,
  "skipped": true,
  "reason": "Frame too blurry (waiting for clear frame)",
  "blur_score": 3.8,
  "predictions": []
}
```

---

### 📊 Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/get_stats` | Aggregate stats: total frames, avg confidence, class breakdown |
| `GET` | `/api/chart_data?limit=500` | Time-series confidence + class data for Chart.js |
| `GET` | `/api/session_scores` | Per-session attention scores for trend chart |
| `GET` | `/api/export?session_id=xxx` | Download detections as CSV |
| `DELETE` | `/api/detections` | Clear all detection records for current user |
| `GET` | `/health` | Health check (no auth required) |

**Stats Response:**
```json
{
  "total_frames": 1240,
  "avg_confidence": 0.887,
  "classes": { "awake": 980, "bored": 180, "sleepy": 80 }
}
```

---

## ⚙️ Configuration Reference

| Variable | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `ROBOFLOW_API_KEY` | ✅ | — | Roboflow account API key |
| `ROBOFLOW_PROJECT` | ✅ | `attention50k` | Roboflow project slug |
| `ROBOFLOW_VERSION` | ✅ | `3` | Model version number |
| `DATABASE_URL` | ✅ | — | PostgreSQL connection string |
| `SECRET_KEY` | ✅ | — | App-level cryptographic secret |
| `JWT_SECRET_KEY` | ✅ | *(SECRET_KEY)* | JWT signing secret |
| `JWT_EXPIRY_HOURS` | ❌ | `24` | Token lifetime in hours |
| `DEBUG` | ❌ | `false` | Enable verbose logging |
| `CONFIDENCE_THRESHOLD` | ❌ | `40` | Min confidence % for valid detection |
| `OVERLAP_THRESHOLD` | ❌ | `30` | NMS overlap % threshold |
| `SMOOTHING_WINDOW` | ❌ | `5` | Frames in temporal voting buffer |
| `ALERT_CONSECUTIVE_FRAMES` | ❌ | `3` | Consecutive inattentive frames before alert |
| `BLUR_THRESHOLD` | ❌ | `15.0` | Laplacian variance rejection threshold |
| `ROBOFLOW_TIMEOUT` | ❌ | `10` | Inference API timeout in seconds |

---

## 🗄️ Database Schema

Tables are automatically created by SQLAlchemy on first startup. No migrations needed.

```sql
-- Users: authenticated accounts
CREATE TABLE users (
    id          SERIAL PRIMARY KEY,
    username    VARCHAR NOT NULL UNIQUE,
    email       VARCHAR NOT NULL UNIQUE,
    password    VARCHAR NOT NULL,          -- bcrypt hash, never plaintext
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Sessions: one row per monitoring session
CREATE TABLE sessions (
    id               VARCHAR PRIMARY KEY,  -- "session-{12 hex chars}"
    user_id          INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    device_id        VARCHAR,              -- optional client fingerprint
    start_time       TIMESTAMP NOT NULL,
    end_time         TIMESTAMP,            -- NULL while session is active
    total_frames     INTEGER DEFAULT 0,    -- frames processed
    attention_score  FLOAT   DEFAULT 0.0   -- % awake frames (0.0 – 100.0)
);

-- Detections: one row per analyzed frame
CREATE TABLE detections (
    id             SERIAL PRIMARY KEY,
    session_id     VARCHAR NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    timestamp      TIMESTAMP NOT NULL,
    class_name     VARCHAR NOT NULL,        -- "awake" | "sleepy" | "bored"
    smoothed_class VARCHAR,                 -- majority-voted class
    confidence     FLOAT   NOT NULL,        -- 0.0 – 1.0
    frame_id       INTEGER                  -- sequential counter per session
);
```

---

## 🔐 Authentication Flow

```
Client                  FastAPI               PostgreSQL
  │                        │                      │
  │  POST /register        │                      │
  │───────────────────────►│                      │
  │  { username, email,    │  INSERT users row     │
  │    password }          │  password=bcrypt(pw) ►│
  │◄───────────────────────│◄─────────────────────│
  │  { success: true }     │                      │
  │                        │                      │
  │  POST /login           │                      │
  │───────────────────────►│  SELECT user WHERE   │
  │  { username, password }│  username=?         ►│
  │                        │◄─────────────────────│
  │                        │  bcrypt.verify(pw)   │
  │◄───────────────────────│                      │
  │  { access_token: JWT } │                      │
  │                        │                      │
  │  POST /api/predict     │                      │
  │  Authorization: Bearer │                      │
  │───────────────────────►│                      │
  │                        │  jwt.decode(token)   │
  │                        │  → extract user_id   │
  │                        │  → process frame     │
  │◄───────────────────────│                      │
  │  { predictions: [...] }│                      │
```

---

## 🚀 Free Cloud Deployment Guide

Deploy the entire stack for **$0/month** using two services.

### Step 1 — Database: Neon.tech

1. Sign up free at **[neon.tech](https://neon.tech)** (no credit card needed).
2. Click **"Create a Project"** → Name it `attentiveness-db`, select **PostgreSQL 17**, pick **AWS US East 1**.
3. Click **"Connect"** → Copy the **Connection String**:
   ```
   postgresql://neondb_owner:<password>@ep-<name>.us-east-1.aws.neon.tech/neondb?sslmode=require
   ```
4. Save it — this is your `DATABASE_URL`. Tables auto-create on first startup.

### Step 2 — App Server: Hugging Face Spaces

1. Sign up free at **[huggingface.co](https://huggingface.co)** (no credit card needed).
2. Click your profile → **New Space** → Select **Docker** SDK.
3. Hardware: **Free (2 vCPU / 16 GB RAM)**. Click **Create Space**.

**Add Secrets** (Settings → Variables and secrets):

| Key | Type | Value |
|-----|------|-------|
| `DATABASE_URL` | 🔒 Secret | Your Neon connection string |
| `ROBOFLOW_API_KEY` | 🔒 Secret | Your Roboflow API key |
| `SECRET_KEY` | 🔒 Secret | Any long random string |
| `JWT_SECRET_KEY` | 🔒 Secret | Any long random string |
| `ROBOFLOW_PROJECT` | Variable | `attention50k` |
| `ROBOFLOW_VERSION` | Variable | `3` |
| `JWT_EXPIRY_HOURS` | Variable | `24` |
| `CONFIDENCE_THRESHOLD` | Variable | `40` |
| `OVERLAP_THRESHOLD` | Variable | `30` |
| `SMOOTHING_WINDOW` | Variable | `5` |
| `ALERT_CONSECUTIVE_FRAMES` | Variable | `3` |
| `BLUR_THRESHOLD` | Variable | `15.0` |
| `DEBUG` | Variable | `false` |

**Push your code:**
```bash
# Get a Write token from: https://huggingface.co/settings/tokens
git remote add huggingface https://<HF_USERNAME>:<HF_TOKEN>@huggingface.co/spaces/<HF_USERNAME>/Attentiveness_Tracker
git push huggingface main
```

Your app will be live at `https://<username>-attentiveness-tracker.hf.space` in ~2 minutes.

---

## 💻 Local Development with Docker Compose

One command brings up three containers: **FastAPI app + PostgreSQL + Nginx proxy**.

### Prerequisites
- [Docker Desktop](https://docs.docker.com/get-docker/) installed and running
- [Roboflow API Key](https://roboflow.com/) (free tier is sufficient)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/sahil8017/Attentiveness-Tracker.git
cd Attentiveness-Tracker

# 2. Set up environment
cp .env.example .env
# Edit .env: fill in ROBOFLOW_API_KEY, SECRET_KEY, JWT_SECRET_KEY

# 3. Build and start all services
docker-compose up --build -d

# 4. View live logs
docker-compose logs -f app

# 5. Stop everything
docker-compose down
```

### Service URLs (local)

| Service | URL |
|---------|-----|
| 🌐 Application | http://localhost:80 |
| 🚀 FastAPI Swagger | http://localhost:5000/docs |
| 📊 FastAPI ReDoc | http://localhost:5000/redoc |
| 🏥 Health Check | http://localhost:5000/health |
| 🗄️ PostgreSQL | localhost:5432 |

---

## 📁 Project Structure

```
attentiveness-tracker/
│
├── main.py                  # FastAPI app: all API routes & detection logic
├── config.py                # Centralized config from environment variables
├── auth.py                  # JWT middleware & dependency helpers
├── db.py                    # SQLAlchemy engine + session factory
├── models.py                # ORM models: User, Session, Detection
│
├── routes/
│   └── auth_routes.py       # /api/auth/* endpoints (register, login, me)
│
├── templates/               # Jinja2 HTML templates
│   ├── index.html           # Landing page
│   ├── detection.html       # Live detection UI
│   ├── dashboard.html       # Analytics dashboard
│   ├── login.html           # Login page
│   └── register.html        # Registration page
│
├── static/
│   ├── css/
│   │   ├── styles.css       # Custom CSS + glassmorphism styles
│   │   └── tailwind.css     # Compiled Tailwind CSS
│   ├── js/
│   │   ├── main.js          # Detection loop, canvas, Chart.js logic
│   │   ├── auth.js          # JWT token management
│   │   └── nav.js           # Navigation & auth state
│   └── alert.mp3            # Audio alert sound file
│
├── Dockerfile               # Production Docker image (port 7860 for HF)
├── docker-compose.yml       # Local dev: app + postgres + nginx
├── nginx/nginx.conf         # Nginx reverse proxy config
├── requirements.txt         # Python dependencies (pinned versions)
├── .env.example             # Template for environment variables
└── README.md                # This file
```

---

## 🤝 Acknowledgements

**AI & Cloud Services:**
- **[Roboflow](https://roboflow.com/)** — RF-DETR Nano model & hosted inference API
- **[Neon.tech](https://neon.tech/)** — Free serverless PostgreSQL
- **[Hugging Face](https://huggingface.co/)** — Free Docker Spaces with 16 GB RAM

**Open-Source Libraries:**
- [FastAPI](https://fastapi.tiangolo.com/) · [OpenCV](https://opencv.org/) · [SQLAlchemy](https://www.sqlalchemy.org/) · [Chart.js](https://www.chartjs.org/) · [Tailwind CSS](https://tailwindcss.com/) · [Lucide Icons](https://lucide.dev/)

**AI Assistants that helped architect, build, and ship this project:**
- **Gemini 3.1 Pro** — Google DeepMind (Antigravity Agent)
- **Claude Sonnet 4.6** — Anthropic

---

<div align="center">

**[🌐 Try the Live App](https://sahil8017-attentiveness-tracker.hf.space/)** &nbsp;·&nbsp; **[⭐ Star on GitHub](https://github.com/sahil8017/Attentiveness-Tracker)** &nbsp;·&nbsp; **[🐛 Report a Bug](https://github.com/sahil8017/Attentiveness-Tracker/issues)**

<br/>

*Built with ❤️ — because staying focused shouldn't require willpower alone.*

</div>
