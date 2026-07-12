"""Cognitive store — SQLite persistence for error fingerprints and confidence logs."""

import sqlite3
import json
import time
from pathlib import Path
from dataclasses import dataclass
from config import DATA_DIR

DB_PATH = DATA_DIR / "cognitive.db"


ERROR_TYPES = [
    "memorization_override",
    "conceptual_gap",
    "terminology_confusion",
    "spelling_sound_conflation",
    "l1_transfer",
    "overgeneralisation",
    "overconfidence",
    "underconfidence",
    "impulsive",
    "shortcut_dependency",
    "fragile_understanding",
    "visualization_weakness",
]


def _conn() -> sqlite3.Connection:
    db = sqlite3.connect(str(DB_PATH))
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.row_factory = sqlite3.Row
    return db


def init_db() -> None:
    db = _conn()
    db.executescript("""
    CREATE TABLE IF NOT EXISTS cognitive_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        topic TEXT NOT NULL DEFAULT '',
        query TEXT NOT NULL,
        student_answer TEXT NOT NULL DEFAULT '',
        correct_answer TEXT NOT NULL DEFAULT '',
        error_type TEXT NOT NULL DEFAULT '',
        error_signature TEXT NOT NULL DEFAULT '',
        explanation TEXT NOT NULL DEFAULT '',
        root_cause_topic TEXT NOT NULL DEFAULT '',
        remediation TEXT NOT NULL DEFAULT '',
        response_ms REAL NOT NULL DEFAULT 0,
        confidence_1to5 INTEGER NOT NULL DEFAULT 0,
        actual_correct INTEGER NOT NULL DEFAULT 0,
        session_id TEXT NOT NULL DEFAULT '',
        created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
    );

    CREATE TABLE IF NOT EXISTS confidence_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        topic TEXT NOT NULL DEFAULT '',
        query TEXT NOT NULL,
        answer TEXT NOT NULL DEFAULT '',
        confidence_1to5 INTEGER NOT NULL DEFAULT 0,
        actual_correct INTEGER NOT NULL DEFAULT 0,
        response_ms REAL NOT NULL DEFAULT 0,
        created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
    );

    CREATE TABLE IF NOT EXISTS cognitive_fingerprints (
        user_id INTEGER PRIMARY KEY,
        fingerprint_json TEXT NOT NULL DEFAULT '{}',
        strengths_json TEXT NOT NULL DEFAULT '[]',
        weaknesses_json TEXT NOT NULL DEFAULT '[]',
        bias_profile_json TEXT NOT NULL DEFAULT '{}',
        topic_coverage_json TEXT NOT NULL DEFAULT '{}',
        resilience_score REAL NOT NULL DEFAULT 0,
        updated_at REAL NOT NULL DEFAULT (strftime('%s','now'))
    );

    CREATE TABLE IF NOT EXISTS error_signatures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signature_hash TEXT UNIQUE NOT NULL,
        error_type TEXT NOT NULL,
        pattern_json TEXT NOT NULL DEFAULT '{}',
        frequency INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
    );

    CREATE INDEX IF NOT EXISTS idx_cog_events_user ON cognitive_events(user_id);
    CREATE INDEX IF NOT EXISTS idx_cog_events_topic ON cognitive_events(topic);
    CREATE INDEX IF NOT EXISTS idx_cog_events_error ON cognitive_events(error_type);
    CREATE INDEX IF NOT EXISTS idx_confidence_user ON confidence_log(user_id);
    """)
    db.commit()
    db.close()


@dataclass
class CognitiveEvent:
    user_id: int
    topic: str
    query: str
    student_answer: str
    correct_answer: str
    error_type: str
    error_signature: str
    explanation: str
    root_cause_topic: str
    remediation: str
    response_ms: float = 0
    confidence_1to5: int = 0
    actual_correct: int = 0
    session_id: str = ""


class CognitiveStore:
    def log_event(self, event: CognitiveEvent) -> int:
        db = _conn()
        cur = db.execute(
            """INSERT INTO cognitive_events
               (user_id, topic, query, student_answer, correct_answer,
                error_type, error_signature, explanation, root_cause_topic,
                remediation, response_ms, confidence_1to5, actual_correct, session_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (event.user_id, event.topic, event.query, event.student_answer,
             event.correct_answer, event.error_type, event.error_signature,
             event.explanation, event.root_cause_topic, event.remediation,
             event.response_ms, event.confidence_1to5, event.actual_correct,
             event.session_id),
        )
        db.commit()
        row_id = cur.lastrowid
        db.close()
        return row_id

    def log_confidence(self, user_id: int, topic: str, query: str,
                       answer: str, confidence_1to5: int,
                       actual_correct: int, response_ms: float) -> None:
        db = _conn()
        db.execute(
            """INSERT INTO confidence_log
               (user_id, topic, query, answer, confidence_1to5,
                actual_correct, response_ms)
               VALUES (?,?,?,?,?,?,?)""",
            (user_id, topic, query, answer, confidence_1to5,
             actual_correct, response_ms),
        )
        db.commit()
        db.close()

    def get_recent_events(self, user_id: int, limit: int = 50) -> list[dict]:
        db = _conn()
        rows = db.execute(
            "SELECT * FROM cognitive_events WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        db.close()
        return [dict(r) for r in rows]

    def error_breakdown(self, user_id: int) -> dict:
        db = _conn()
        rows = db.execute(
            "SELECT error_type, COUNT(*) as cnt FROM cognitive_events "
            "WHERE user_id=? GROUP BY error_type ORDER BY cnt DESC",
            (user_id,),
        ).fetchall()
        db.close()
        return {r["error_type"]: r["cnt"] for r in rows}

    def aggregate_error_breakdown(self, user_ids: list[int]) -> dict:
        """School-wide misconception tally across many students (real errors only)."""
        if not user_ids:
            return {}
        ph = ",".join("?" for _ in user_ids)
        db = _conn()
        rows = db.execute(
            f"SELECT error_type, COUNT(*) as cnt FROM cognitive_events "
            f"WHERE user_id IN ({ph}) AND error_type NOT IN ('', 'no_error') "
            f"GROUP BY error_type ORDER BY cnt DESC",
            tuple(user_ids),
        ).fetchall()
        db.close()
        return {r["error_type"]: r["cnt"] for r in rows}

    def confidence_calibration(self, user_id: int) -> dict:
        db = _conn()
        rows = db.execute(
            """SELECT confidence_1to5, actual_correct, COUNT(*) as cnt
               FROM confidence_log
               WHERE user_id=?
               GROUP BY confidence_1to5, actual_correct
               ORDER BY confidence_1to5""",
            (user_id,),
        ).fetchall()
        db.close()
        stats: dict = {}
        for r in rows:
            level = r["confidence_1to5"]
            if level not in stats:
                stats[level] = {"correct": 0, "wrong": 0, "total": 0}
            if r["actual_correct"]:
                stats[level]["correct"] += r["cnt"]
            else:
                stats[level]["wrong"] += r["cnt"]
            stats[level]["total"] += r["cnt"]
        for level, s in stats.items():
            s["accuracy"] = round(s["correct"] / max(1, s["total"]) * 100, 1)
        return stats

    def save_fingerprint(self, user_id: int, fingerprint_json: dict) -> None:
        db = _conn()
        db.execute(
            """INSERT OR REPLACE INTO cognitive_fingerprints
               (user_id, fingerprint_json, strengths_json, weaknesses_json,
                bias_profile_json, topic_coverage_json, resilience_score, updated_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                user_id,
                json.dumps(fingerprint_json.get("summary", {})),
                json.dumps(fingerprint_json.get("strengths", [])),
                json.dumps(fingerprint_json.get("weaknesses", [])),
                json.dumps(fingerprint_json.get("biases", {})),
                json.dumps(fingerprint_json.get("topics", {})),
                fingerprint_json.get("resilience_score", 0),
                time.time(),
            ),
        )
        db.commit()
        db.close()

    def load_fingerprint(self, user_id: int) -> dict | None:
        db = _conn()
        row = db.execute(
            "SELECT * FROM cognitive_fingerprints WHERE user_id=?", (user_id,)
        ).fetchone()
        db.close()
        if not row:
            return None
        return {
            "user_id": row["user_id"],
            "fingerprint": json.loads(row["fingerprint_json"]),
            "strengths": json.loads(row["strengths_json"]),
            "weaknesses": json.loads(row["weaknesses_json"]),
            "biases": json.loads(row["bias_profile_json"]),
            "topics": json.loads(row["topic_coverage_json"]),
            "resilience_score": row["resilience_score"],
            "updated_at": row["updated_at"],
        }

    def topic_weakness_map(self, user_id: int) -> dict:
        db = _conn()
        rows = db.execute(
            "SELECT topic, error_type, COUNT(*) as cnt FROM cognitive_events "
            "WHERE user_id=? AND error_type IN ('conceptual_gap','shortcut_dependency',"
            "'fragile_understanding','memorization_override') "
            "GROUP BY topic, error_type ORDER BY topic, cnt DESC",
            (user_id,),
        ).fetchall()
        db.close()
        result: dict = {}
        for r in rows:
            topic = r["topic"]
            if topic not in result:
                result[topic] = {}
            result[topic][r["error_type"]] = r["cnt"]
        return result


store = CognitiveStore()
