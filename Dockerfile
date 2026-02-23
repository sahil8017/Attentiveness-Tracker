# ==========================================
# Attentiveness Tracker — Dockerfile
# Multi-stage build for minimal image size
# ==========================================

FROM python:3.11-slim AS base

# Install OpenCV system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0t64 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py database.py main.py ./
COPY static/ ./static/
COPY templates/ ./templates/

# Create required directories
RUN mkdir -p logs temp static/images

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:5000/health'); r.raise_for_status()" || exit 1

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1", "--access-log"]
