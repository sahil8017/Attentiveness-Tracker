---
title: Attentiveness Tracker
emoji: 📊
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
---

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=30&pause=1000&color=6200EE&center=true&vCenter=true&width=600&lines=🎯+Attentiveness+Tracker;Real-Time+AI+Focus+Monitor;Know+When+You+Drift%2C+Stay+Sharp." alt="Typing SVG" />

<br/>

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Headless-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Roboflow](https://img.shields.io/badge/Roboflow-RF--DETR_Nano-6200EE?style=for-the-badge)](https://roboflow.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Neon](https://img.shields.io/badge/Neon-Serverless_DB-00E599?style=for-the-badge)](https://neon.tech/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue?style=for-the-badge)](https://huggingface.co/spaces/sahil8017/Attentiveness_Tracker)

<br/>

> **A production-grade, AI-powered real-time focus monitoring system.** Analyzes live webcam feeds every second and classifies your attentiveness state as **Awake**, **Sleepy**, or **Bored** — then alerts you before you lose focus entirely.

<br/>

**[🌐 Live Demo →](https://sahil8017-attentiveness-tracker.hf.space/)** | **[📖 API Docs →](https://sahil8017-attentiveness-tracker.hf.space/docs)** | **[⚙️ GitHub →](https://github.com/sahil8017/Attentiveness-Tracker)**

</div>

---

## 📖 Table of Contents

- [💡 What Is This?](#-what-is-this)
- [✨ Key Features](#-key-features)
- [🏗️ System Architecture](#️-system-architecture)
- [🛠️ Tech Stack](#️-tech-stack)
- [🔁 How It Works — The Detection Pipeline](#-how-it-works--the-detection-pipeline)
- [📡 REST API Reference](#-rest-api-reference)
- [⚙️ Configuration Reference](#️-configuration-reference)
- [🚀 Free Cloud Deployment Guide](#-free-cloud-deployment-guide)
- [💻 Local Development (Docker Compose)](#-local-development-docker-compose)
- [🗄️ Database Schema](#️-database-schema)
- [🔐 Authentication Flow](#-authentication-flow)
- [🤝 Acknowledgements](#-acknowledgements)

---

## 💡 What Is This?

**Attentiveness Tracker** is a production-grade web application that uses a state-of-the-art computer vision model to monitor your level of focus in real time. It works entirely in your browser — no software installation required — by accessing your webcam and continuously analyzing frames.

Built for:
- 📚 **Students** who need to stay focused during long study sessions.
- 💼 **Remote workers** attending video calls or doing deep work.
- 🧑‍🏫 **Educators** who want to monitor classroom engagement.
- 🔬 **Researchers** studying attention spans and cognitive load.

It is built on a robust, asynchronous Python backend, a serverless PostgreSQL database, and is deployed on a completely **free**, modern cloud infrastructure.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| ⚡ **Real-time Detection** | Processes live webcam frames at ~15–30 FPS via async httpx requests to the Roboflow Inference API. |
| 🛡️ **Temporal Smoothing** | A majority-vote algorithm over a rolling buffer of N recent frames eliminates flickering between states — giving stable, reliable classifications. |
| 🔍 **Frame Quality Gating** | Rejects blurry frames using OpenCV Laplacian variance before sending to the model, preventing noisy predictions. |
| 🔔 **Smart Audio Alerts** | Alerts fire only after `N` consecutive inattentive frames, not on a single detection — no more false alarms. |
| 📊 **Interactive Analytics Dashboard** | Live Chart.js visualizations including confidence trend lines, class distributions, and historical per-session attention scores. |
| 🔐 **Full JWT Authentication** | Multi-user system with bcrypt-hashed passwords and signed JWT tokens. Every session and detection is securely scoped to the authenticated user. |
| 📱 **Responsive Design** | Glassmorphism-inspired UI with auto Dark/Light mode, fully responsive from mobile to desktop. |
| ☁️ **Serverless & Free** | Entire production stack runs for $0/month using Hugging Face Spaces (2 vCPU / 16GB RAM) and Neon.tech Serverless Postgres. |
| 🐳 **Docker-Ready** | Single-command local setup with Docker Compose (FastAPI + PostgreSQL + Nginx). |
| 📁 **CSV Export** | Export any session's raw detection data as a CSV file for offline analysis. |

---

## 🏗️ System Architecture

The application follows a clean, layered architecture optimized for async throughput.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER'S WEB BROWSER                              │
│                                                                         │
│  ┌─────────────┐    Base64 JPEG    ┌─────────────────────────────────┐  │
│  │  Webcam     │ ──────────────►   │   JavaScript Frontend Engine    │  │
│  │  (HTML5)    │                   │   (Canvas API + Chart.js +      │  │
│  └─────────────┘                   │    Lucide + Tailwind CSS)       │  │
│                                    └──────────────┬──────────────────┘  │
└───────────────────────────────────────────────────┼─────────────────────┘
                                                    │ HTTPS POST /api/predict
                                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                🤗  HUGGING FACE SPACES  (Docker Container)              │
│                         2 vCPU / 16 GB RAM / FREE                      │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    FastAPI Application                           │   │
│  │                                                                  │   │
│  │  ① JWT Auth Middleware  ──► Validates Bearer Token               │   │
│  │  ② OpenCV Engine        ──► Laplacian blur detection             │   │
│  │  ③ Roboflow Client      ──► Async httpx POST to Inference API    │   │
│  │  ④ Temporal Smoother    ──► Majority vote over rolling buffer    │   │
│  │  ⑤ Alert Engine         ──► Consecutive inattentive frame count  │   │
│  │  ⑥ SQLAlchemy ORM       ──► Write Detection + Update Session     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────┬───────────────────┘
                               │                      │
                               ▼                      ▼
         ┌─────────────────────────────┐    ┌─────────────────────────┐
         │    🤖 ROBOFLOW CLOUD         │    │   🗄️ NEON.TECH          │
         │    RF-DETR Nano Model        │    │   Serverless PostgreSQL  │
         │    (attention50k v3)         │    │   (Free Tier)           │
         │    Hosted Inference API      │    │                         │
         └─────────────────────────────┘    └─────────────────────────┘
```

---

## 🛠️ Tech Stack

### ⚙️ Backend
| Component | Technology | Version |
|-----------|-----------|---------|
| Web Framework | FastAPI | 0.115.0 |
| ASGI Server | Uvicorn (with `standard` extras) | 0.30.0 |
| HTTP Client | httpx (async) | 0.27.0 |
| ORM | SQLAlchemy | 2.0.35 |
| Database Driver | psycopg2-binary | 2.9.9 |
| Template Engine | Jinja2 | 3.1.4 |
| Computer Vision | OpenCV (Headless) | 4.10.0 |
| Image Processing | Pillow | 10.4.0 |
| Numerical | NumPy | 1.26.4 |
| Auth | python-jose + bcrypt | 3.3.0 / 4.2.1 |
| Validation | Pydantic (with email) | 2.9.2 |
| Config | python-dotenv | 1.0.1 |

### 🎨 Frontend
| Component | Technology |
|-----------|-----------|
| Styling | Tailwind CSS (CDN + custom config) |
| Data Visualization | Chart.js |
| Icons | Lucide Icons |
| Webcam Access | HTML5 `getUserMedia` API |
| Frame Capture | HTML5 Canvas API (off-screen rendering) |

### ☁️ Infrastructure
| Component | Service | Cost |
|-----------|---------|------|
| Application Hosting | Hugging Face Spaces (Docker) | **$0 / month** |
| Database | Neon.tech Serverless PostgreSQL 17 | **$0 / month** |
| AI Inference | Roboflow Hosted API (RF-DETR Nano) | **$0 / month** |
| Container Registry | Docker (via HF Spaces build) | **$0 / month** |
| **Total** | | **$0 / month** |

---

## 🔁 How It Works — The Detection Pipeline

Here is the complete end-to-end lifecycle of a single frame being processed, from webcam pixel to database record:

```
STEP 1: CAPTURE
  Browser captures a frame from the webcam every ~100ms
  using an offscreen HTML5 Canvas element.
  The frame is compressed to a Base64-encoded JPEG string.

STEP 2: AUTHENTICATE
  The Base64 frame + the current Session ID are sent as a
  JSON POST to /api/predict with a JWT Bearer token in the header.
  The FastAPI JWT middleware validates the token. If invalid → 401.

STEP 3: DECODE
  The server base64-decodes the image data into raw bytes.
  OpenCV (cv2.imdecode) converts the bytes into a NumPy array (BGR matrix).

STEP 4: QUALITY GATE
  OpenCV computes the Laplacian variance of the grayscale frame.
  If variance < BLUR_THRESHOLD (15.0), the frame is blurry.
    → Returns last known good state to the client (no API call made).

STEP 5: INFERENCE
  The base64 image data is sent as a form body to the Roboflow
  Inference REST API endpoint via an async httpx POST request.
  The RF-DETR Nano model returns bounding boxes, class names, and confidence scores.

STEP 6: TEMPORAL SMOOTHING
  For each detected class, it is appended to a rolling deque buffer of
  size SMOOTHING_WINDOW (5 frames) for that session.
  The majority class across the buffer becomes the "smoothed_class".
  This prevents rapid flickering between Awake/Sleepy states.

STEP 7: ALERT ENGINE
  If the smoothed class is "sleepy" or "bored", an alert counter increments.
  If the counter reaches ALERT_CONSECUTIVE_FRAMES (3), the response flags
  trigger_alert: true, and the frontend plays a Web Audio API beep.
  Any attentive frame resets the counter to 0.

STEP 8: PERSIST
  A Detection record is written to the PostgreSQL database via SQLAlchemy,
  storing: session_id, timestamp, class_name, confidence, frame_id, smoothed_class.
  The parent Session record's attention_score and total_frames are updated in real-time.

STEP 9: RESPOND
  The server returns JSON with predictions, smoothed_class, blur_score,
  frame_id, and the trigger_alert flag back to the browser in ~50–200ms.

STEP 10: RENDER
  The JavaScript frontend draws bounding boxes on the live video canvas,
  updates the on-screen stat counters, appends a data point to the Chart.js
  confidence graph, and triggers an audio alert if flagged.
```

---

## 📡 REST API Reference

All API endpoints (except `/health`) require a JWT Bearer token in the `Authorization` header.

**Base URL:** `https://sahil8017-attentiveness-tracker.hf.space`

### 🔑 Authentication

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|:---:|
| `POST` | `/api/auth/register` | Register a new user account | ❌ |
| `POST` | `/api/auth/login` | Login and receive a JWT access token | ❌ |
| `GET`  | `/api/auth/me` | Get the current authenticated user's profile | ✅ |

#### `POST /api/auth/register`
```json
// Request Body
{ "username": "sahil", "email": "sahil@example.com", "password": "SecurePass123" }

// Response 200
{ "success": true, "message": "User registered successfully", "user_id": 1 }
```

#### `POST /api/auth/login`
```json
// Request Body
{ "username": "sahil", "password": "SecurePass123" }

// Response 200
{ "access_token": "eyJhbGci...", "token_type": "bearer", "expires_in": 86400 }
```

---

### 🎬 Session Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/sessions` | Create a new tracking session |
| `POST` | `/api/sessions/{id}/end` | End a session and compute final attention score |
| `GET`  | `/api/sessions` | List all sessions for the current user |
| `DELETE` | `/api/sessions/{id}` | Delete a specific session |
| `DELETE` | `/api/sessions` | Delete **all** sessions for the current user |

#### `POST /api/sessions`
```json
// Request Body
{ "device_id": "optional-client-fingerprint" }

// Response 200
{ "success": true, "session_id": "session-09d151943dc4" }
```

---

### 🧠 Prediction

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/predict` | Submit a frame for attentiveness analysis |

#### `POST /api/predict`
```json
// Request Body
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgAB...",
  "session_id": "session-09d151943dc4"
}

// Response 200 — Successful Detection
{
  "success": true,
  "predictions": [
    {
      "x": 320, "y": 240, "width": 150, "height": 200,
      "class": "awake",
      "smoothed_class": "awake",
      "confidence": 0.94
    }
  ],
  "frame_id": 42,
  "blur_score": 87.3,
  "trigger_alert": false
}

// Response 200 — Blurry Frame Skipped
{
  "success": true,
  "skipped": true,
  "reason": "Frame too blurry (waiting for clear frame)",
  "blur_score": 4.1,
  "predictions": []
}
```

---

### 📊 Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/get_stats` | Aggregate stats (total frames, avg confidence, class breakdown) |
| `GET` | `/api/chart_data` | Time-series data for Chart.js (last 500 detections) |
| `GET` | `/api/session_scores` | Attention scores per session for the trend chart |
| `GET` | `/api/export` | Export detection data as a downloadable CSV file |
| `GET` | `/health` | Health check endpoint (no auth) |

---

## ⚙️ Configuration Reference

All configuration is loaded from environment variables at startup. Set these in your `.env` file for local development, or as **Secrets** in your cloud provider's dashboard.

| Variable | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `ROBOFLOW_API_KEY` | ✅ | — | Your personal Roboflow API key for model inference. |
| `ROBOFLOW_PROJECT` | ✅ | `attention50k` | Roboflow project slug (namespace). |
| `ROBOFLOW_VERSION` | ✅ | `3` | Version number of the Roboflow model to use. |
| `DATABASE_URL` | ✅ | — | Full PostgreSQL connection string (`postgresql://...`). |
| `SECRET_KEY` | ✅ | — | Master cryptographic secret for the application. |
| `JWT_SECRET_KEY` | ✅ | *(falls back to `SECRET_KEY`)* | Secret used to sign and verify JWT tokens. |
| `JWT_EXPIRY_HOURS` | ❌ | `24` | How long (in hours) a login session lasts before expiry. |
| `DEBUG` | ❌ | `false` | Enables verbose logging and FastAPI debug mode. |
| `CONFIDENCE_THRESHOLD` | ❌ | `40` | Minimum confidence % for a detection to be considered valid. |
| `OVERLAP_THRESHOLD` | ❌ | `30` | Maximum bounding box overlap % before non-max suppression. |
| `SMOOTHING_WINDOW` | ❌ | `5` | Number of recent frames used in the temporal majority-vote smoothing buffer. |
| `ALERT_CONSECUTIVE_FRAMES` | ❌ | `3` | Number of consecutive inattentive predictions required to trigger an audio alert. |
| `BLUR_THRESHOLD` | ❌ | `15.0` | Laplacian variance threshold. Frames below this value are rejected as too blurry. |
| `ROBOFLOW_TIMEOUT` | ❌ | `10` | Timeout in seconds for requests to the Roboflow Inference API. |

---

## 🚀 Free Cloud Deployment Guide

This project is designed to run at **$0/month** using two free-tier services.

### Step 1: Database — Neon.tech

1. Create a free account at **[neon.tech](https://neon.tech)**.
2. Click **"Create a new project"**.
3. Set a project name (e.g., `attentiveness-db`), pick **PostgreSQL 17**, and choose the region closest to your app server (e.g., `AWS US East 1`).
4. Once created, click **"Connect"** and copy the full **Connection String**. It looks like:
   ```
   postgresql://neondb_owner:<password>@ep-<name>.us-east-1.aws.neon.tech/neondb?sslmode=require
   ```
5. Save this — it's your `DATABASE_URL`.

> **Note:** Neon's free tier includes **0.5 GB storage**, **branching**, and auto-pause of the compute when idle. Your database is automatically created and tables are provisioned by SQLAlchemy on first startup.

---

### Step 2: Application Server — Hugging Face Spaces

1. Create a free account at **[huggingface.co](https://huggingface.co)**.
2. Click your profile → **New Space**.
3. Configure the Space:
   - **Space name:** `Attentiveness_Tracker`
   - **SDK:** Click **Docker**
   - **Hardware:** Select **Free (2 vCPU / 16 GB RAM)**
   - **Visibility:** Public
4. Click **Create Space**.

#### Add all Secrets (Settings → Variables and secrets):

| Key | Type | Value |
|-----|------|-------|
| `DATABASE_URL` | **Secret** | Your Neon connection string |
| `ROBOFLOW_API_KEY` | **Secret** | Your Roboflow API key |
| `SECRET_KEY` | **Secret** | A random secure string |
| `JWT_SECRET_KEY` | **Secret** | A random secure string |
| `ROBOFLOW_PROJECT` | Variable | `attention50k` |
| `ROBOFLOW_VERSION` | Variable | `3` |
| `JWT_EXPIRY_HOURS` | Variable | `24` |
| `CONFIDENCE_THRESHOLD` | Variable | `40` |
| `OVERLAP_THRESHOLD` | Variable | `30` |
| `SMOOTHING_WINDOW` | Variable | `5` |
| `ALERT_CONSECUTIVE_FRAMES` | Variable | `3` |
| `BLUR_THRESHOLD` | Variable | `15.0` |
| `DEBUG` | Variable | `false` |

#### Push your code:

```bash
# Add the Hugging Face remote
git remote add huggingface https://<YOUR_HF_USERNAME>:<YOUR_HF_TOKEN>@huggingface.co/spaces/<YOUR_HF_USERNAME>/Attentiveness_Tracker

# Push!
git push huggingface main
```

> Your app will be live in ~2 minutes at:
> `https://<your-username>-attentiveness-tracker.hf.space`

---

## 💻 Local Development (Docker Compose)

Running locally spins up **three containers** — the FastAPI app, a local PostgreSQL database, and an Nginx reverse proxy — all in one command.

### Prerequisites
- [Docker Desktop](https://docs.docker.com/get-docker/) installed and running.
- A [Roboflow API Key](https://roboflow.com/) (free tier is sufficient).

### Setup Steps

**1. Clone the repository:**
```bash
git clone https://github.com/sahil8017/Attentiveness-Tracker.git
cd Attentiveness-Tracker
```

**2. Set up your environment file:**
```bash
cp .env.example .env
```
Now open `.env` and fill in your values:
```env
ROBOFLOW_API_KEY=your_roboflow_key_here
DATABASE_URL=postgresql://attentiveness:attentiveness_dev@db:5432/attentiveness
SECRET_KEY=any-long-random-secret-string
JWT_SECRET_KEY=another-long-random-secret-string
```

**3. Build and start all containers:**
```bash
docker-compose up --build -d
```

**4. Access the application:**

| Service | URL |
|---------|-----|
| Application | `http://localhost:80` |
| FastAPI Docs (Swagger) | `http://localhost:5000/docs` |
| FastAPI ReDoc | `http://localhost:5000/redoc` |
| PostgreSQL | `localhost:5432` |

**5. Stop and remove containers:**
```bash
docker-compose down
```

**6. View live application logs:**
```bash
docker-compose logs -f app
```

---

## 🗄️ Database Schema

All data is managed by SQLAlchemy and automatically provisioned on startup.

```sql
-- Users table: stores registered accounts
CREATE TABLE users (
    id          SERIAL PRIMARY KEY,
    username    VARCHAR NOT NULL UNIQUE,
    email       VARCHAR NOT NULL UNIQUE,
    password    VARCHAR NOT NULL,              -- bcrypt hash
    created_at  TIMESTAMP DEFAULT now()
);

-- Sessions table: one record per monitoring session
CREATE TABLE sessions (
    id               VARCHAR PRIMARY KEY,      -- e.g. "session-09d151943dc4"
    user_id          INTEGER REFERENCES users(id) ON DELETE CASCADE,
    device_id        VARCHAR,                  -- optional client fingerprint
    start_time       TIMESTAMP NOT NULL,
    end_time         TIMESTAMP,
    total_frames     INTEGER DEFAULT 0,
    attention_score  FLOAT DEFAULT 0.0         -- % of "awake" frames
);

-- Detections table: one record per analyzed frame
CREATE TABLE detections (
    id             SERIAL PRIMARY KEY,
    session_id     VARCHAR REFERENCES sessions(id) ON DELETE CASCADE,
    timestamp      TIMESTAMP NOT NULL,
    class_name     VARCHAR NOT NULL,           -- raw class: "awake", "sleepy", "bored"
    smoothed_class VARCHAR,                    -- majority-vote smoothed class
    confidence     FLOAT NOT NULL,             -- e.g. 0.94
    frame_id       INTEGER                     -- sequential frame counter per session
);
```

---

## 🔐 Authentication Flow

```
  User              Frontend              FastAPI            PostgreSQL
   │                   │                    │                    │
   │  POST /register   │                    │                    │
   │──────────────────►│                    │                    │
   │                   │  POST /api/auth/register               │
   │                   │───────────────────►│                    │
   │                   │                    │  INSERT user row   │
   │                   │                    │───────────────────►│
   │                   │                    │  (password = bcrypt hash)
   │                   │◄───────────────────│                    │
   │◄──────────────────│                    │                    │
   │   { success: true }                    │                    │
   │                   │                    │                    │
   │  POST /login      │                    │                    │
   │──────────────────►│                    │                    │
   │                   │  POST /api/auth/login                  │
   │                   │───────────────────►│                    │
   │                   │                    │  SELECT user       │
   │                   │                    │───────────────────►│
   │                   │                    │  bcrypt.verify()   │
   │                   │◄───────────────────│                    │
   │◄──────────────────│                    │                    │
   │  { access_token: "eyJ..." }            │                    │
   │                   │                    │                    │
   │  [Stores JWT in   │                    │                    │
   │   localStorage]   │                    │                    │
   │                   │                    │                    │
   │  POST /api/predict│                    │                    │
   │  + Authorization: Bearer eyJ...        │                    │
   │──────────────────►│───────────────────►│                    │
   │                   │                    │  Validate JWT      │
   │                   │                    │  → Extract user_id │
   │                   │                    │  → Process frame   │
```

---

## 🤝 Acknowledgements

This project stands on the shoulders of incredible open-source tools and AI systems.

**AI Models & Services:**
- **[Roboflow](https://roboflow.com/)** — For their RF-DETR Nano model and the hosted inference API that powers the core detection capability.
- **[Neon.tech](https://neon.tech/)** — For the generous free-tier serverless PostgreSQL service.
- **[Hugging Face](https://huggingface.co/)** — For the incredibly powerful free Docker Spaces with 16GB RAM.

**Open-Source Libraries:**
- **[FastAPI](https://fastapi.tiangolo.com/)** — The blazing-fast async Python web framework.
- **[OpenCV](https://opencv.org/)** — The industry-standard computer vision library.
- **[SQLAlchemy](https://www.sqlalchemy.org/)** — The Python SQL toolkit and ORM.
- **[Chart.js](https://www.chartjs.org/)** — Beautiful, responsive data visualizations.
- **[Tailwind CSS](https://tailwindcss.com/)** — Utility-first CSS framework for rapid UI development.

**AI Assistants that helped architect, build, refactor, and ship this application:**
- **Gemini 3.1 Pro** (Google DeepMind — Antigravity Agent)
- **Claude Sonnet 4.6** (Anthropic)

---

<div align="center">

**[🌐 Try the Live App](https://sahil8017-attentiveness-tracker.hf.space/)** · **[⭐ Star this repo](https://github.com/sahil8017/Attentiveness-Tracker)** · **[🐛 Report a Bug](https://github.com/sahil8017/Attentiveness-Tracker/issues)**

<br/>

*Built with ❤️ — because staying focused shouldn't require willpower alone.*

</div>
