<div align="center">
  
# 🎯 Attentiveness Tracker

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Roboflow](https://img.shields.io/badge/Roboflow-6200EE?style=for-the-badge&logo=roboflow&logoColor=white)](https://roboflow.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

An AI-powered, real-time focus monitoring application that analyzes webcam feeds to track attentiveness states (**Awake**, **Sleepy**, or **Bored**). Built for students, professionals, and anyone looking to optimize their productivity through actionable insights.

</div>

---

## ✨ Outstanding Features

- **⚡ Real-time Detection:** Millisecond-level inference using the Roboflow Inference API (RF-DETR Nano model).
- **🛡️ Temporal Smoothing:** Advanced algorithmic voting across consecutive frames completely eliminates erratic flickering and false positive detections.
- **📱 True Cross-Device Support:** Dynamic camera orientation ensures perfect aspect ratios and zero "squishing" on mobile devices and variable screens.
- **🔔 Smart Audio Alerts:** Configurable, non-intrusive alerts that trigger *only* after sustained periods of inattention to keep you focused.
- **📊 Interactive Analytics:** Beautiful Chart.js dashboards showing live confidence trends, class distributions, and historical session scoring.
- **🔐 Secure Authentication:** JWT-based user authentication securely tracking individual sessions, backed by a robust PostgreSQL database.
- **🐳 Production Ready:** Fully containerized with Docker and Docker Compose for instant, reproducible deployments across any environment.

## 🛠️ Technology Stack

### Backend Core
- **Framework:** FastAPI (High-performance, async Python web framework)
- **Database:** PostgreSQL (Relational database management)
- **ORM:** SQLAlchemy 2.0
- **Authentication:** JWT (JSON Web Tokens) & bcrypt hashing
- **Computer Vision:** OpenCV (Headless)

### Frontend Engine
- **Design:** Modern Glassmorphism with Auto Dark/Light mode support
- **Styling:** Tailwind CSS
- **Data Visualization:** Chart.js
- **Icons:** Lucide Icons

### AI & Infrastructure
- **Model:** Roboflow (RF-DETR Nano)
- **Containerization:** Docker & Docker Compose
- **Server:** Uvicorn ASGI

---

## 🚀 Quick Start Guide

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) & Docker Compose
- [Roboflow API Key](https://roboflow.com/) (Free tier available)

### Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sahil8017/Attentiveness-Tracker.git
   cd Attentiveness-Tracker
   ```

2. **Configure your environment:**
   ```bash
   cp .env.example .env
   ```
   *Open `.env` and add your `ROBOFLOW_API_KEY`, along with any custom database credentials.*

3. **Launch the application:**
   ```bash
   docker-compose up --build -d
   ```

4. **Access the application:**
   Open your browser and navigate to `http://localhost:8000`.

---

## ⚙️ Configuration Reference

All core settings are easily managed via the `.env` configuration file:

| Variable | Description | Default |
|----------|-------------|---------|
| `ROBOFLOW_API_KEY` | **Required.** Your API key for model inference. | — |
| `DATABASE_URL` | PostgreSQL connection string. | `postgresql://user:pass@db:5432/attentiveness` |
| `SECRET_KEY` | Secret key used for cryptographic JWT signing. | *Generate a secure hash* |
| `CONFIDENCE_THRESHOLD` | Minimum percentage confidence required for a valid bounding box detection. | `40` |
| `SMOOTHING_WINDOW` | Number of frames used for temporal voting (reduces noise significantly). | `5` |
| `ALERT_CONSECUTIVE_FRAMES` | Sustained frames of inattention required before an alert fires. | `3` |

---

## 🏗️ System Architecture

The application is built on a high-performance asynchronous, non-blocking architecture. 
1. The frontend client captures video frames using extremely optimized, dynamic HTML5 off-screen Canvas elements, safely transmitting compressed JPEG buffers to the FastAPI backend. 
2. The Python backend orchestrates the Roboflow Vision API predictions while applying a tailored temporal smoothing algorithm to the results.
3. The backend logs the session metrics securely to PostgreSQL, and returns the actionable bounding box and classification data back to the client for rendering at ~15-30 FPS.

---

## 🤝 Special Acknowledgements

This final version marks the successful conclusion of development. A massive thank you to the AI assistants that helped architect, build, refactor, and scale this application from a simple Flask script into a production-grade FastAPI monolith:

- **Claude 3.7 Sonnet** 
- **Gemini 1.5 Pro**
- **Antigravity (Google DeepMind)**

---

<div align="center">
  <i>Built with ❤️ for better focus and elevated productivity.</i>
</div>
