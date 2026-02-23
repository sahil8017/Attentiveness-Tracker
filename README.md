# 🎯 Attentiveness Tracker

AI-powered real-time attentiveness detection using webcam analysis. Tracks your focus state — **awake**, **sleepy**, or **bored** — with temporal smoothing, session management, and interactive analytics.

**Live Demo:** Uses Roboflow Inference API (RF-DETR Nano model) for fast, accurate detection.

## ✨ Features

- **Real-time Detection** — Live webcam feed with instant classification and bounding boxes
- **Temporal Smoothing** — Majority vote across frames reduces noise for reliable results
- **Smart Alerts** — Alerts only trigger after sustained inattention (configurable threshold)
- **Analytics Dashboard** — Chart.js powered with confidence trends, class distribution, session history
- **Session Management** — Create, track, and compare multiple detection sessions
- **CSV Export** — Download detection data for external analysis
- **Docker Ready** — Development and production Docker Compose configs included

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + Uvicorn |
| AI Model | Roboflow Inference API (RF-DETR Nano) |
| Computer Vision | OpenCV |
| HTTP Client | httpx (async) |
| Database | SQLite |
| Frontend | HTML + Tailwind CSS + Chart.js |
| Deployment | Docker Compose + Nginx |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+ **or** Docker
- [Roboflow API Key](https://roboflow.com/) (free tier available)

### Option 1: Local Development

```bash
# Clone the repo
git clone https://github.com/your-username/attentiveness-tracker.git
cd attentiveness-tracker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set your ROBOFLOW_API_KEY

# Run the app
python main.py
```

Open **http://localhost:5000** in your browser.

### Option 2: Docker Compose (Development)

```bash
# Configure environment
cp .env.example .env
# Edit .env and set your ROBOFLOW_API_KEY

# Build and run
docker compose up --build

# Run in background
docker compose up -d --build
```

Open **http://localhost:5000** in your browser.

### Option 3: Docker Compose (Production)

```bash
# Configure environment
cp .env.example .env
# Edit .env — set ROBOFLOW_API_KEY and DEBUG=false

# Build and deploy with Nginx
docker compose -f docker-compose.prod.yml up -d --build
```

Open **http://localhost** (port 80) in your browser.

## 📁 Project Structure

```
attentiveness-tracker/
├── main.py                    # FastAPI application (entry point)
├── config.py                  # Centralized configuration
├── database.py                # SQLite database module
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Dev Docker Compose
├── docker-compose.prod.yml    # Production Docker Compose (with Nginx)
├── .env.example               # Environment variable template
├── nginx/
│   └── nginx.conf             # Nginx reverse proxy config
├── static/
│   ├── css/styles.css         # Global styles
│   ├── js/main.js             # Detection client script
│   └── alert.mp3              # Alert sound
└── templates/
    ├── index.html             # Homepage
    ├── detection.html         # Live detection page
    └── dashboard.html         # Analytics dashboard
```

## ⚙️ Configuration

All settings are in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ROBOFLOW_API_KEY` | — | **Required.** Your Roboflow API key |
| `ROBOFLOW_PROJECT` | `attention50k` | Roboflow project name |
| `ROBOFLOW_VERSION` | `3` | Model version |
| `CONFIDENCE_THRESHOLD` | `40` | Min confidence % for detections |
| `BLUR_THRESHOLD` | `15.0` | Laplacian variance threshold for blur |
| `SMOOTHING_WINDOW` | `5` | Frames for temporal smoothing |
| `ALERT_CONSECUTIVE_FRAMES` | `3` | Inattentive frames before alert |
| `DEBUG` | `false` | Enable debug mode |

## 🌐 Deployment Options

### Cloud Platforms

| Platform | How |
|----------|-----|
| **Railway** | Connect GitHub repo → auto-deploys with Dockerfile |
| **Render** | Connect repo → select Docker runtime → set env vars |
| **Fly.io** | `fly launch` → `fly deploy` |
| **Google Cloud Run** | `gcloud run deploy --source .` |
| **AWS ECS** | Push to ECR → create ECS service |
| **DigitalOcean App Platform** | Connect repo → auto-detect Dockerfile |

### Self-Hosted (VPS/VM)

```bash
# SSH into your server
ssh user@your-server

# Clone and deploy
git clone https://github.com/your-username/attentiveness-tracker.git
cd attentiveness-tracker
cp .env.example .env
# Edit .env with your API key

# Production deployment with Nginx
docker compose -f docker-compose.prod.yml up -d --build

# View logs
docker compose -f docker-compose.prod.yml logs -f
```

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Send frame for detection |
| `POST` | `/api/sessions` | Create session |
| `POST` | `/api/sessions/{id}/end` | End session |
| `GET` | `/api/sessions` | List sessions |
| `DELETE` | `/api/sessions/{id}` | Delete session |
| `DELETE` | `/api/sessions` | Clear all sessions |
| `GET` | `/get_stats` | Get statistics |
| `GET` | `/api/chart_data` | Chart.js data |
| `GET` | `/api/session_scores` | Session trends |
| `GET` | `/api/export` | Export CSV |

## 📝 License

MIT License — feel free to use and modify.

---

Built with ❤️ using FastAPI, Roboflow AI, and OpenCV.
