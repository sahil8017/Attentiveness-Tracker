"""
SQLite database module for Attentiveness Tracker.
Handles session and detection data storage with indexed queries.
"""
import sqlite3
import csv
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager


class Database:
    """SQLite database manager for attentiveness data."""

    def __init__(self, db_path):
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    total_frames INTEGER DEFAULT 0,
                    attention_score REAL DEFAULT 0.0
                );

                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    frame_id INTEGER NOT NULL,
                    smoothed_class TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE INDEX IF NOT EXISTS idx_detections_session
                    ON detections(session_id);
                CREATE INDEX IF NOT EXISTS idx_detections_timestamp
                    ON detections(timestamp);
                CREATE INDEX IF NOT EXISTS idx_detections_class
                    ON detections(class);
            """)

    @contextmanager
    def _get_conn(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # === Session Operations ===

    def create_session(self, session_id):
        """Create a new detection session."""
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO sessions (id, start_time) VALUES (?, ?)",
                (session_id, datetime.now().isoformat())
            )

    def end_session(self, session_id):
        """End a detection session and compute attention score."""
        with self._get_conn() as conn:
            # Calculate attention score
            row = conn.execute(
                """SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN class = 'awake' THEN 1 ELSE 0 END) as attentive
                FROM detections WHERE session_id = ?""",
                (session_id,)
            ).fetchone()

            total = row["total"] if row["total"] else 0
            attentive = row["attentive"] if row["attentive"] else 0
            score = round((attentive / total * 100), 1) if total > 0 else 0.0

            conn.execute(
                """UPDATE sessions 
                SET end_time = ?, total_frames = ?, attention_score = ?
                WHERE id = ?""",
                (datetime.now().isoformat(), total, score, session_id)
            )
            return score

    def get_sessions(self, limit=20):
        """Get recent sessions."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM sessions 
                ORDER BY start_time DESC LIMIT ?""",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_session(self, session_id):
        """Get a specific session."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            return dict(row) if row else None

    # === Detection Operations ===

    def log_detection(self, session_id, class_name, confidence, frame_id, smoothed_class=None):
        """Log a single detection result."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO detections 
                (session_id, timestamp, class, confidence, frame_id, smoothed_class)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, datetime.now().isoformat(), class_name,
                 round(confidence, 3), frame_id, smoothed_class)
            )

    # === Analytics Queries ===

    def get_stats(self, session_id=None):
        """Get aggregate statistics, optionally filtered by session."""
        with self._get_conn() as conn:
            where = "WHERE session_id = ?" if session_id else ""
            params = (session_id,) if session_id else ()

            row = conn.execute(f"""
                SELECT 
                    COUNT(*) as total_frames,
                    ROUND(AVG(confidence), 3) as avg_confidence
                FROM detections {where}
            """, params).fetchone()

            class_rows = conn.execute(f"""
                SELECT class, COUNT(*) as count
                FROM detections {where}
                GROUP BY class ORDER BY count DESC
            """, params).fetchall()

            return {
                "total_frames": row["total_frames"],
                "avg_confidence": row["avg_confidence"] or 0,
                "classes": {r["class"]: r["count"] for r in class_rows}
            }

    def get_chart_data(self, session_id=None, limit=500):
        """Get time-series data formatted for Chart.js."""
        with self._get_conn() as conn:
            where = "WHERE session_id = ?" if session_id else ""
            params = (session_id, limit) if session_id else (limit,)

            rows = conn.execute(f"""
                SELECT timestamp, class, confidence, smoothed_class, frame_id
                FROM detections {where}
                ORDER BY timestamp DESC LIMIT ?
            """, params).fetchall()

            # Reverse to chronological order
            data = [dict(r) for r in reversed(rows)]

            return {
                "labels": [d["timestamp"] for d in data],
                "confidence": [d["confidence"] for d in data],
                "classes": [d["smoothed_class"] or d["class"] for d in data],
                "frame_ids": [d["frame_id"] for d in data]
            }

    def get_session_scores(self, limit=20):
        """Get attention scores across sessions for trend chart."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT id, start_time, attention_score, total_frames
                FROM sessions WHERE end_time IS NOT NULL
                ORDER BY start_time DESC LIMIT ?""",
                (limit,)
            ).fetchall()
            return [dict(r) for r in reversed(rows)]

    def get_detections_for_export(self, session_id=None):
        """Get all detections for CSV/JSON export."""
        with self._get_conn() as conn:
            where = "WHERE d.session_id = ?" if session_id else ""
            params = (session_id,) if session_id else ()

            rows = conn.execute(f"""
                SELECT d.timestamp, d.class, d.confidence, d.frame_id,
                       d.smoothed_class, d.session_id
                FROM detections d {where}
                ORDER BY d.timestamp ASC
            """, params).fetchall()
            return [dict(r) for r in rows]

    # === Delete Operations ===

    def delete_session(self, session_id):
        """Delete a specific session and all its detections."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM detections WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    def clear_all_sessions(self):
        """Delete ALL sessions and their detections."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM detections")
            conn.execute("DELETE FROM sessions")

    def clear_all_detections(self):
        """Delete ALL detection records and reset session stats."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM detections")
            conn.execute(
                "UPDATE sessions SET total_frames = 0, attention_score = 0.0"
            )

    # === Migration ===

    def migrate_from_csv(self, csv_path, session_id="migrated"):
        """Import existing CSV log data into the database."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            return 0

        count = 0
        with self._get_conn() as conn:
            # Skip if already migrated
            existing = conn.execute(
                "SELECT id FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if existing:
                return 0

            # Create a migration session
            conn.execute(
                "INSERT INTO sessions (id, start_time) VALUES (?, ?)",
                (session_id, datetime.now().isoformat())
            )

            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        conn.execute(
                            """INSERT INTO detections 
                            (session_id, timestamp, class, confidence, frame_id)
                            VALUES (?, ?, ?, ?, ?)""",
                            (session_id, row.get("Time", ""),
                             row.get("Class", ""), float(row.get("Confidence", 0)),
                             int(row.get("Frame_ID", 0)))
                        )
                        count += 1
                    except (ValueError, KeyError):
                        continue

            # Compute session stats inline (avoid nested connection)
            if count > 0:
                row = conn.execute(
                    """SELECT COUNT(*) as total,
                        SUM(CASE WHEN class = 'awake' THEN 1 ELSE 0 END) as attentive
                    FROM detections WHERE session_id = ?""",
                    (session_id,)
                ).fetchone()
                total = row["total"] if row["total"] else 0
                attentive = row["attentive"] if row["attentive"] else 0
                score = round((attentive / total * 100), 1) if total > 0 else 0.0
                conn.execute(
                    """UPDATE sessions 
                    SET end_time = ?, total_frames = ?, attention_score = ?
                    WHERE id = ?""",
                    (datetime.now().isoformat(), total, score, session_id)
                )

        return count

