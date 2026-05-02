"""
Database engine and session factory for Attentiveness Tracker.
PostgreSQL only — configured via DATABASE_URL environment variable.
"""

import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def init_db(database_url: str):
    """
    Initialize PostgreSQL database engine and create tables.
    Call once at app startup.
    """
    global _engine, _SessionLocal

    _engine = create_engine(
        database_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )

    Base.metadata.create_all(bind=_engine)
    logger.info("PostgreSQL database initialized, tables verified.")

    _SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=_engine,
    )


def get_db():
    """
    FastAPI dependency: yields a database session, auto-closes after request.
    Usage:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()
