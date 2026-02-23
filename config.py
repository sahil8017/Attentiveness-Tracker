"""
Centralized configuration for Attentiveness Tracker.
Loads settings from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Application configuration."""

    # Base directory
    BASE_DIR = BASE_DIR

    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "attentiveness-tracker-secret-key-change-me")
    DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

    # Roboflow API
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
    ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT", "attention50k")
    ROBOFLOW_VERSION = int(os.getenv("ROBOFLOW_VERSION", "3"))

    # Database
    DATABASE_PATH = BASE_DIR / os.getenv("DATABASE_NAME", "attentiveness.db")

    # Detection Settings
    CONFIDENCE_THRESHOLD = int(os.getenv("CONFIDENCE_THRESHOLD", "50"))
    OVERLAP_THRESHOLD = int(os.getenv("OVERLAP_THRESHOLD", "30"))
    SMOOTHING_WINDOW = int(os.getenv("SMOOTHING_WINDOW", "5"))  # Number of frames for temporal smoothing
    ALERT_CONSECUTIVE_FRAMES = int(os.getenv("ALERT_CONSECUTIVE_FRAMES", "3"))  # Consecutive inattentive frames before alert
    BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "15.0"))  # Laplacian variance threshold for blur detection

    # Roboflow Inference API (direct HTTP instead of SDK)
    ROBOFLOW_API_URL = os.getenv(
        "ROBOFLOW_API_URL",
        "https://detect.roboflow.com"
    )
    ROBOFLOW_TIMEOUT = int(os.getenv("ROBOFLOW_TIMEOUT", "10"))  # seconds

    # Paths
    PLOTS_DIR = BASE_DIR / "static" / "images"
    LOGS_DIR = BASE_DIR / "logs"
    TEMP_DIR = BASE_DIR / "temp"

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.ROBOFLOW_API_KEY:
            raise ValueError(
                "ROBOFLOW_API_KEY is required. Set it in .env file or as an environment variable."
            )

        # Ensure directories exist
        cls.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)
