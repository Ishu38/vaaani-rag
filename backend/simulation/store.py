"""Simulation store — SQLite persistence for exam simulation sessions."""

import sqlite3
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from config import DATA_DIR

DB_PATH = DATA_DIR / "simulation.db"


def _conn() -> sqlite3.Connection:
    db = sqlite3.connect(str(DB_PATH))
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.row_factory = sqlite3.Row
    return db


def init_db() -> None:
    db = _conn()
    db.executescript("""
    CREATE TABLE IF NOT EXISTS simulation_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_id TEXT UNIQUE NOT NULL,
        subject TEXT NOT NULL DEFAULT '',
        config_json TEXT NOT NULL DEFAULT '{}',
        started_at REAL NOT NULL DEFAULT (strftime('%s','now')),
        completed_at REAL,
        total_questions INTEGER NOT NULL DEFAULT 0,
        attempted INTEGER NOT NULL DEFAULT 0,
        correct INTEGER NOT NULL DEFAULT 0,
        wrong INTEGER NOT NULL DEFAULT 0,
        skipped INTEGER NOT NULL DEFAULT 0,
        total_score REAL NOT NULL DEFAULT 0,
        max_score REAL NOT NULL DEFAULT 0,
        avg_response_ms REAL NOT NULL DEFAULT 0,
        stress_resilience_score REAL NOT NULL DEFAULT 0,
        state TEXT NOT NULL DEFAULT 'idle'
    );

    CREATE TABLE IF NOT EXISTS simulation_answers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        user_id INTEGER NOT NULL,
        question_index INTEGER NOT NULL,
        topic TEXT NOT NULL DEFAULT '',
        difficulty REAL NOT NULL DEFAULT 1.0,
        query TEXT NOT NULL,
        correct_answer TEXT NOT NULL,
        student_answer TEXT NOT NULL DEFAULT '',
        was_correct INTEGER NOT NULL DEFAULT 0,
        response_ms REAL NOT NULL DEFAULT 0,
        confidence_1to5 INTEGER NOT NULL DEFAULT 0,
        confidence_0to100 INTEGER NOT NULL DEFAULT -1,
        coaching_interjection TEXT NOT NULL DEFAULT '',
        pressure_state_json TEXT NOT NULL DEFAULT '{}',
        is_flagged INTEGER NOT NULL DEFAULT 0,
        peer_session_id TEXT NOT NULL DEFAULT '',
        created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
        FOREIGN KEY (session_id) REFERENCES simulation_sessions(session_id)
    );

    CREATE TABLE IF NOT EXISTS simulation_analytics (
        user_id INTEGER PRIMARY KEY,
        stress_resilience REAL NOT NULL DEFAULT 0.5,
        time_vs_accuracy_tradeoff_json TEXT NOT NULL DEFAULT '{}',
        recovery_rate REAL NOT NULL DEFAULT 0.5,
        impulsive_tendency REAL NOT NULL DEFAULT 0,
        best_topic TEXT NOT NULL DEFAULT '',
        worst_topic TEXT NOT NULL DEFAULT '',
        sessions_completed INTEGER NOT NULL DEFAULT 0,
        updated_at REAL NOT NULL DEFAULT (strftime('%s','now'))
    );

    CREATE INDEX IF NOT EXISTS idx_sim_sessions_user ON simulation_sessions(user_id);
    CREATE INDEX IF NOT EXISTS idx_sim_answers_session ON simulation_answers(session_id);
    CREATE INDEX IF NOT EXISTS idx_sim_answers_user ON simulation_answers(user_id);
    """)
    db.commit()
    db.close()


class SimulationStore:
    def create_session(self, session_id: str, user_id: int, subject: str,
                       config: dict) -> None:
        db = _conn()
        db.execute(
            """INSERT INTO simulation_sessions
               (user_id, session_id, subject, config_json, state)
               VALUES (?,?,?,?,?)""",
            (user_id, session_id, subject, json.dumps(config), "running"),
        )
        db.commit()
        db.close()

    def get_session(self, session_id: str) -> dict | None:
        db = _conn()
        row = db.execute(
            "SELECT * FROM simulation_sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        db.close()
        return dict(row) if row else None

    def update_session_state(self, session_id: str, state: str) -> None:
        db = _conn()
        db.execute(
            "UPDATE simulation_sessions SET state=? WHERE session_id=?",
            (state, session_id),
        )
        db.commit()
        db.close()

    def complete_session(self, session_id: str, stats: dict) -> None:
        db = _conn()
        db.execute(
            """UPDATE simulation_sessions SET
               completed_at=?, total_questions=?, attempted=?, correct=?,
               wrong=?, skipped=?, total_score=?, max_score=?,
               avg_response_ms=?, stress_resilience_score=?, state='complete'
               WHERE session_id=?""",
            (
                time.time(), stats.get("total_questions", 0),
                stats.get("attempted", 0), stats.get("correct", 0),
                stats.get("wrong", 0), stats.get("skipped", 0),
                stats.get("total_score", 0), stats.get("max_score", 0),
                stats.get("avg_response_ms", 0),
                stats.get("stress_resilience", 0),
                session_id,
            ),
        )
        db.commit()
        db.close()

    def log_answer(self, session_id: str, user_id: int, question_index: int,
                   topic: str, difficulty: float, query: str,
                   correct_answer: str, student_answer: str,
                   was_correct: int, response_ms: float, confidence_1to5: int,
                   coaching_interjection: str, pressure_state: dict,
                   is_flagged: int, confidence_0to100: int = -1) -> None:
        db = _conn()
        db.execute(
            """INSERT INTO simulation_answers
               (session_id, user_id, question_index, topic, difficulty,
                query, correct_answer, student_answer, was_correct,
                response_ms, confidence_1to5, confidence_0to100,
                coaching_interjection, pressure_state_json, is_flagged)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (session_id, user_id, question_index, topic, difficulty,
             query, correct_answer, student_answer, was_correct,
             response_ms, confidence_1to5, confidence_0to100,
             coaching_interjection, json.dumps(pressure_state), is_flagged),
        )
        db.commit()
        db.close()

    def get_answers(self, session_id: str) -> list[dict]:
        db = _conn()
        rows = db.execute(
            "SELECT * FROM simulation_answers WHERE session_id=? ORDER BY question_index",
            (session_id,),
        ).fetchall()
        db.close()
        return [dict(r) for r in rows]

    def get_sessions(self, user_id: int, limit: int = 20) -> list[dict]:
        db = _conn()
        rows = db.execute(
            """SELECT * FROM simulation_sessions
               WHERE user_id=? AND state='complete'
               ORDER BY completed_at DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        db.close()
        return [dict(r) for r in rows]

    def update_analytics(self, user_id: int, analytics: dict) -> None:
        db = _conn()
        db.execute(
            """INSERT OR REPLACE INTO simulation_analytics
               (user_id, stress_resilience, time_vs_accuracy_tradeoff_json,
                recovery_rate, impulsive_tendency, best_topic, worst_topic,
                sessions_completed, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                user_id,
                analytics.get("stress_resilience", 0.5),
                json.dumps(analytics.get("time_vs_accuracy", {})),
                analytics.get("recovery_rate", 0.5),
                analytics.get("impulsive_tendency", 0),
                analytics.get("best_topic", ""),
                analytics.get("worst_topic", ""),
                analytics.get("sessions_completed", 0),
                time.time(),
            ),
        )
        db.commit()
        db.close()

    def get_analytics(self, user_id: int) -> dict | None:
        db = _conn()
        row = db.execute(
            "SELECT * FROM simulation_analytics WHERE user_id=?", (user_id,)
        ).fetchone()
        db.close()
        if not row:
            return None
        r = dict(row)
        r["time_vs_accuracy"] = json.loads(r["time_vs_accuracy_tradeoff_json"])
        return r


sim_store = SimulationStore()
