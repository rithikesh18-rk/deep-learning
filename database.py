import sqlite3
import pandas as pd
from datetime import datetime
import os
import logging
from config import DB_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_connection():
    """Establish connection to SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize SQLite database schema for helmet detection history."""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS DetectionHistory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    helmet_count INTEGER NOT NULL DEFAULT 0,
                    without_helmet_count INTEGER NOT NULL DEFAULT 0,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    result TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

def save_detection(filename: str, file_type: str, helmet_count: int, without_helmet_count: int, confidence: float, result: str):
    """Save detection result record to database."""
    try:
        init_db()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO DetectionHistory 
                (filename, file_type, helmet_count, without_helmet_count, confidence, result, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (filename, file_type, helmet_count, without_helmet_count, round(confidence, 4), result, now_str))
            conn.commit()
            logger.info(f"Detection logged for {filename}")
            return cursor.lastrowid
    except Exception as e:
        logger.error(f"Error saving detection to DB: {e}")
        return None

def get_history() -> pd.DataFrame:
    """Fetch all detection history records as pandas DataFrame."""
    try:
        init_db()
        with get_connection() as conn:
            query = "SELECT id, filename, file_type, helmet_count, without_helmet_count, confidence, result, timestamp FROM DetectionHistory ORDER BY timestamp DESC"
            df = pd.read_sql_query(query, conn)
            return df
    except Exception as e:
        logger.error(f"Error reading detection history: {e}")
        return pd.DataFrame(columns=[
            "id", "filename", "file_type", "helmet_count", 
            "without_helmet_count", "confidence", "result", "timestamp"
        ])

def delete_history():
    """Clear all records from DetectionHistory table."""
    try:
        init_db()
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM DetectionHistory")
            conn.commit()
            logger.info("Detection history cleared.")
            return True
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        return False

def get_analytics_summary() -> dict:
    """Calculate overall statistics for dashboard analytics."""
    try:
        init_db()
        df = get_history()
        if df.empty:
            return {
                "total_detections": 0,
                "helmet_count": 0,
                "without_helmet_count": 0,
                "avg_confidence": 0.0,
                "compliance_rate": 0.0
            }
        
        total_detections = len(df)
        total_helmets = int(df["helmet_count"].sum())
        total_without_helmets = int(df["without_helmet_count"].sum())
        avg_conf = float(df["confidence"].mean()) if not df["confidence"].isna().all() else 0.0
        
        total_people = total_helmets + total_without_helmets
        compliance_rate = round((total_helmets / total_people * 100), 2) if total_people > 0 else 0.0

        return {
            "total_detections": total_detections,
            "helmet_count": total_helmets,
            "without_helmet_count": total_without_helmets,
            "avg_confidence": round(avg_conf * 100, 1),
            "compliance_rate": compliance_rate
        }
    except Exception as e:
        logger.error(f"Error calculating analytics summary: {e}")
        return {
            "total_detections": 0,
            "helmet_count": 0,
            "without_helmet_count": 0,
            "avg_confidence": 0.0,
            "compliance_rate": 0.0
        }

if __name__ == "__main__":
    init_db()
    print("Database test passed!")
