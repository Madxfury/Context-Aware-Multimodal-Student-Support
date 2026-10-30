import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "counsellor.db")

class DatabaseManager:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = os.path.abspath(db_path or os.environ.get("DATABASE_PATH", DB_PATH))
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Users table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                role TEXT,
                age INTEGER,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Sessions table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                locked_emotion TEXT,
                confidence REAL,
                mental_health_score REAL,
                risk_level TEXT,
                primary_concern TEXT,
                interventions_used TEXT,
                stress_trajectory TEXT,
                average_hr REAL,
                transcript TEXT,
                summary_insight TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """)
            conn.commit()

    def get_or_create_user(self, name: str, role: Optional[str] = None, age: Optional[int] = None, notes: Optional[str] = None) -> Dict[str, Any]:
        name_clean = name.strip()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE LOWER(name) = LOWER(?)", (name_clean,))
            row = cursor.fetchone()
            if row:
                user = dict(row)
                # Update last seen and any updated fields
                cursor.execute("""
                    UPDATE users 
                    SET last_seen_at = CURRENT_TIMESTAMP,
                        role = COALESCE(?, role),
                        age = COALESCE(?, age),
                        notes = COALESCE(?, notes)
                    WHERE id = ?
                """, (role, age, notes, user["id"]))
                conn.commit()
                return self.get_user_by_id(user["id"])
            else:
                cursor.execute("""
                    INSERT INTO users (name, role, age, notes, created_at, last_seen_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (name_clean, role or "Student", age, notes or ""))
                conn.commit()
                user_id = cursor.lastrowid
                return self.get_user_by_id(user_id)

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_users(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, role, age, notes, last_seen_at FROM users ORDER BY last_seen_at DESC LIMIT 20")
            return [dict(row) for row in cursor.fetchall()]

    def save_session(self, session_data: Dict[str, Any]) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            transcript_json = json.dumps(session_data.get("transcript", []))
            interventions_json = json.dumps(session_data.get("interventions_used", []))
            stress_trajectory_json = json.dumps(session_data.get("stress_trajectory", []))
            
            cursor.execute("""
                INSERT OR REPLACE INTO sessions (
                    session_id, user_id, locked_emotion, confidence,
                    mental_health_score, risk_level, primary_concern,
                    interventions_used, stress_trajectory, average_hr,
                    transcript, summary_insight, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM sessions WHERE session_id = ?), CURRENT_TIMESTAMP))
            """, (
                session_data.get("session_id"),
                session_data.get("user_id"),
                session_data.get("locked_emotion"),
                session_data.get("confidence"),
                session_data.get("mental_health_score"),
                session_data.get("risk_level"),
                session_data.get("primary_concern"),
                interventions_json,
                stress_trajectory_json,
                session_data.get("average_hr"),
                transcript_json,
                session_data.get("summary_insight"),
                session_data.get("session_id")
            ))
            
            # Update user's last seen
            if session_data.get("user_id"):
                cursor.execute("UPDATE users SET last_seen_at = CURRENT_TIMESTAMP WHERE id = ?", (session_data.get("user_id"),))
                
            conn.commit()
            return True

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.*, u.name as user_name, u.role as user_role, u.age as user_age
                FROM sessions s
                LEFT JOIN users u ON s.user_id = u.id
                WHERE s.session_id = ?
            """, (session_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            try:
                data["transcript"] = json.loads(data["transcript"]) if data["transcript"] else []
            except Exception:
                data["transcript"] = []
            try:
                data["interventions_used"] = json.loads(data["interventions_used"]) if data["interventions_used"] else []
            except Exception:
                data["interventions_used"] = []
            try:
                data["stress_trajectory"] = json.loads(data["stress_trajectory"]) if data["stress_trajectory"] else []
            except Exception:
                data["stress_trajectory"] = []
            return data

    def get_user_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, created_at, locked_emotion, confidence,
                       mental_health_score, risk_level, primary_concern,
                       summary_insight, average_hr
                FROM sessions
                WHERE user_id = ?
                ORDER BY created_at DESC
            """, (user_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_user_recent_context(self, user_id: int, limit: int = 3) -> str:
        """
        Retrieves summaries of the user's recent sessions formatted
        as background memory for the AI Counselor.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, created_at, locked_emotion, mental_health_score, 
                       risk_level, primary_concern, summary_insight, transcript
                FROM sessions
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit))
            rows = cursor.fetchall()
            
            if not rows:
                return "This is the user's first recorded counselling check-in. Welcome them warmly."
                
            context_blocks = []
            for row in rows:
                r = dict(row)
                date_str = r.get("created_at", "Previous Date")
                try:
                    dt = datetime.fromisoformat(date_str)
                    date_str = dt.strftime("%b %d, %Y at %I:%M %p")
                except Exception:
                    pass
                    
                # Extract last user message or brief snippet
                transcript = []
                if r.get("transcript"):
                    try:
                        transcript = json.loads(r["transcript"])
                    except Exception:
                        pass
                
                user_msgs = [m.get("content") for m in transcript if m.get("role") == "user"]
                last_topic = user_msgs[-1] if user_msgs else (r.get("primary_concern") or "General check-in")
                
                block = f"- Session on {date_str}: Emotion detected: {r.get('locked_emotion')}, Mental Health Score: {r.get('mental_health_score')}/100 ({r.get('risk_level')}). Topic/Concern: \"{last_topic}\"."
                if r.get("summary_insight"):
                    block += f" Counselor Insight: {r.get('summary_insight')}"
                context_blocks.append(block)
                
            return "PAST SESSION HISTORY & LONG-TERM MEMORY:\n" + "\n".join(context_blocks) + "\n\nUse this context to acknowledge previous interactions, show continuity of care, and warmly follow up on their progress where natural."
