"""
Database engine and session factory for Attentiveness Tracker.
Supports SQLite (dev) and PostgreSQL (production) via DATABASE_URL.
"""

import logging
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def init_db(database_url: str):
    """
    Initialize database engine and create tables.
    Call once at app startup.

    If the existing database has stale tables (missing columns from
    the new schema), they are dropped and recreated to avoid
    OperationalError on INSERT.
    """
    global _engine, _SessionLocal

    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    _engine = create_engine(
        database_url,
        connect_args=connect_args,
        echo=False,
        pool_pre_ping=True,
    )

    # --- Schema migration: detect stale tables ---
    try:
        inspector = inspect(_engine)
        existing_tables = inspector.get_table_names()

        # Check if 'sessions' table exists but is missing new columns
        if "sessions" in existing_tables:
            columns = {col["name"] for col in inspector.get_columns("sessions")}
            required = {"user_id", "device_id"}
            missing = required - columns
            if missing:
                logger.warning(
                    f"Stale 'sessions' table detected (missing: {missing}). "
                    "Dropping old tables and recreating with new schema."
                )
                Base.metadata.drop_all(bind=_engine)

        # Check if 'users' table is missing entirely (pre-auth database)
        if "sessions" in existing_tables and "users" not in existing_tables:
            logger.warning("Pre-auth database detected. Dropping old tables.")
            Base.metadata.drop_all(bind=_engine)

    except Exception as e:
        logger.warning(f"Schema inspection failed (will attempt create_all): {e}")

    # Create all tables (new or after drop)
    Base.metadata.create_all(bind=_engine)
    logger.info("Database tables verified/created successfully.")

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
