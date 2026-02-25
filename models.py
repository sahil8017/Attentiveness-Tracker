"""
SQLAlchemy ORM Models for Attentiveness Tracker.
Supports SQLite (dev) and PostgreSQL (production).
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime,
    ForeignKey, Index, Boolean
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}')>"


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String(64), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    device_id = Column(String(64), nullable=True, index=True)
    start_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    total_frames = Column(Integer, default=0)
    attention_score = Column(Float, default=0.0)

    # Relationships
    user = relationship("User", back_populates="sessions")
    detections = relationship("Detection", back_populates="session", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "device_id": self.device_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_frames": self.total_frames,
            "attention_score": self.attention_score,
        }


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("sessions.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    class_name = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    frame_id = Column(Integer, nullable=False)
    smoothed_class = Column(String(50), nullable=True)

    # Relationships
    session = relationship("Session", back_populates="detections")

    __table_args__ = (
        Index("idx_detections_session", "session_id"),
        Index("idx_detections_timestamp", "timestamp"),
        Index("idx_detections_class", "class_name"),
    )

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "class": self.class_name,
            "confidence": self.confidence,
            "frame_id": self.frame_id,
            "smoothed_class": self.smoothed_class,
            "session_id": self.session_id,
        }
