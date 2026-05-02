---
title: Attentiveness Tracker
emoji: 📊
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
---

<div align="center">
  
# 🎯 Attentiveness Tracker

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Roboflow](https://img.shields.io/badge/Roboflow-6200EE?style=for-the-badge&logo=roboflow&logoColor=white)](https://roboflow.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue?style=for-the-badge)](https://huggingface.co/spaces)
[![Neon](https://img.shields.io/badge/Neon-00E599?style=for-the-badge&logo=neon&logoColor=black)](https://neon.tech/)
[![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

An AI-powered, real-time focus monitoring application that analyzes webcam feeds to track attentiveness states (**Awake**, **Sleepy**, or **Bored**). Built for students, professionals, and anyone looking to optimize their productivity through actionable insights.

</div>

---

## 🌟 Live Demo

The application is deployed live in a production environment using a completely free architecture:
👉 **[View the Live Application](https://sahil8017-attentiveness-tracker.hf.space/)**

---

## ✨ Outstanding Features

- **⚡ Real-time Detection:** Millisecond-level inference using the Roboflow Inference API (RF-DETR Nano model).
- **🛡️ Temporal Smoothing:** Advanced algorithmic voting across consecutive frames completely eliminates erratic flickering and false positive detections.
- **📱 True Cross-Device Support:** Dynamic camera orientation ensures perfect aspect ratios and zero "squishing" on mobile devices and variable screens.
- **🔔 Smart Audio Alerts:** Configurable, non-intrusive alerts that trigger *only* after sustained periods of inattention to keep you focused.
- **📊 Interactive Analytics:** Beautiful Chart.js dashboards showing live confidence trends, class distributions, and historical session scoring.
- **🔐 Secure Authentication:** JWT-based user authentication securely tracking individual sessions, backed by a robust PostgreSQL database.
- **☁️ Cloud-Native Architecture:** Fully containerized Docker environment utilizing Hugging Face Spaces for compute and Neon.tech for serverless databases.

---

## 🛠️ Technology Stack

### Backend Core
- **Framework:** FastAPI (High-performance, async Python web framework)
- **Database:** PostgreSQL (Neon.tech Serverless DB)
- **ORM:** SQLAlchemy 2.0
- **Authentication:** JWT (JSON Web Tokens) & bcrypt hashing
- **Computer Vision:** OpenCV (Headless) & Roboflow SDK

### Frontend Engine
- **Design:** Modern Glassmorphism with Auto Dark/Light mode support
- **Styling:** Tailwind CSS
- **Data Visualization:** Chart.js
- **Icons:** Lucide Icons

---

## 🏗️ System Architecture

```mermaid
graph TD
    Client[Web Browser Client] -->|Captures Frame| Canvas(HTML5 Canvas)
    Canvas -->|Base64 JPEG| FastAPI[FastAPI Server]
    
    subgraph Hugging Face Spaces [Hugging Face Spaces (Docker Container)]
        FastAPI -->|Async Request| Auth[JWT Auth Middleware]
        FastAPI -->|Check Blur| CV[OpenCV Quality Engine]
        FastAPI -->|Smooth Data| Smoothing[Temporal Voting Buffer]
    end
    
    FastAPI <-->|REST API| Roboflow[Roboflow RF-DETR Nano Model]
    
    subgraph Neon [Neon.tech Serverless DB]
        DB[(PostgreSQL Database)]
    end
    
    FastAPI <-->|SQLAlchemy ORM| DB
```

---

## ⚙️ Configuration Reference

All core settings are managed via environment variables. In production, these should be added as Secrets/Variables.

| Variable | Description | Example |
|----------|-------------|---------|
| `ROBOFLOW_API_KEY` | **Required.** Your API key for model inference. | `CRO2...` |
| `ROBOFLOW_PROJECT` | Roboflow project namespace. | `attention50k` |
| `DATABASE_URL` | PostgreSQL connection string. | `postgresql://user:pass@ep-....aws.neon.tech/neondb` |
| `SECRET_KEY` | Secret key used for cryptographic JWT signing. | *Generate a secure hash* |
| `CONFIDENCE_THRESHOLD` | Minimum percentage confidence required for a valid bounding box. | `40` |
| `SMOOTHING_WINDOW` | Number of frames used for temporal voting (reduces noise). | `5` |
| `ALERT_CONSECUTIVE_FRAMES` | Sustained frames of inattention required before an alert fires. | `3` |
| `BLUR_THRESHOLD` | Laplacian variance threshold to reject blurry frames. | `15.0` |

---

## 🚀 Deployment Guide (100% Free Architecture)

This application is designed to be deployed for **free** using modern serverless platforms. 

### 1. Database (Neon.tech)
1. Create a free account at [Neon.tech](https://neon.tech).
2. Create a new project (e.g., `attentiveness-db`) in the closest region to your server (e.g., `US East`).
3. Copy the **Connection String** provided.

### 2. Application Server (Hugging Face Spaces)
1. Create a new **Docker Space** on [Hugging Face](https://huggingface.co/spaces).
2. Select the **Free (2 vCPU / 16 GB RAM)** tier (The 16GB RAM is highly recommended for OpenCV applications).
3. In the Space **Settings > Variables and secrets**, add all the required environment variables.
4. Clone the space locally, add this project's code, and push to the Hugging Face remote repository.

---

## 💻 Local Setup (Development)

If you wish to run the project locally, Docker Compose makes it a one-step process.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sahil8017/Attentiveness-Tracker.git
   cd Attentiveness-Tracker
   ```

2. **Configure your environment:**
   ```bash
   cp .env.example .env
   ```
   *Open `.env` and add your `ROBOFLOW_API_KEY`.*

3. **Launch the application:**
   ```bash
   docker-compose up --build -d
   ```

4. **Access the application:**
   Open your browser and navigate to `http://localhost:5000`.

---

<div align="center">
  <i>Built with ❤️ for better focus and elevated productivity.</i>
</div>
