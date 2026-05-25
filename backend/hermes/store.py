"""SQLite-backed trace log for the Hermes self-correcting layer.

One row per /chat call. Embeddings stored as raw float32 BLOBs; k-NN is a
brute-force cosine scan over up to a few thousand rows — fast enough for a
personal RAG and avoids dragging in another vector index for this side-path.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from auth.db import connect

_LOCK = threading.Lock()
_INITIALIZED = False


HERMES_SCHEMA = """
-- One row per /chat interaction. user_id is NULL for anonymous sessions.
CREATE TABLE IF NOT EXISTS hermes_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,                              -- NULL for anonymous
    query TEXT NOT NULL,
    embedding BLOB NOT NULL,                      -- float32 vector
    intent TEXT NOT NULL,                         -- knowledge|task|calendar|meta
    graph_mode TEXT,                              -- local|global|NULL
    num_chunks INTEGER NOT NULL DEFAULT 0,
    fidelity_warnings INTEGER NOT NULL DEFAULT 0, -- count of flagged sentences
    tokens INTEGER NOT NULL DEFAULT 0,
    corrections_applied TEXT NOT NULL DEFAULT '[]', -- JSON list of correction names
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_hermes_user_time ON hermes_traces(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_hermes_intent ON hermes_traces(intent);
"""


def init_hermes_db() -> None:
    """Idempotent — create the hermes_traces table on first import."""
    global _INITIALIZED
    with _LOCK:
        if _INITIALIZED:
            return
        with connect() as c:
            c.executescript(HERMES_SCHEMA)
        _INITIALIZED = True


@dataclass
class Trace:
    """A past /chat interaction with the metadata Hermes needs to learn from."""
    id: int
    user_id: int | None
    query: str
    embedding: np.ndarray
    intent: str
    graph_mode: str | None
    num_chunks: int
    fidelity_warnings: int
    tokens: int
    corrections_applied: list[str]
    created_at: str
    similarity: float = 0.0   # filled in by nearest_traces()


def _emb_to_blob(v: np.ndarray) -> bytes:
    """Pack a numpy vector as float32 bytes for BLOB storage."""
    return np.asarray(v, dtype=np.float32).tobytes()


def _blob_to_emb(b: bytes) -> np.ndarray:
    """Unpack a float32 BLOB back into a 1-D numpy vector."""
    return np.frombuffer(b, dtype=np.float32)


def _row_to_trace(row: sqlite3.Row) -> Trace:
    """SQLite row → Trace dataclass."""
    return Trace(
        id=row["id"],
        user_id=row["user_id"],
        query=row["query"],
        embedding=_blob_to_emb(row["embedding"]),
        intent=row["intent"],
        graph_mode=row["graph_mode"],
        num_chunks=row["num_chunks"],
        fidelity_warnings=row["fidelity_warnings"],
        tokens=row["tokens"],
        corrections_applied=json.loads(row["corrections_applied"] or "[]"),
        created_at=row["created_at"],
    )


def log_trace(
    *,
    user_id: int | None,
    query: str,
    embedding: np.ndarray,
    intent: str,
    graph_mode: str | None,
    num_chunks: int,
    fidelity_warnings: int,
    tokens: int,
    corrections_applied: list[str],
) -> int:
    """Persist one trace; return the new row id."""
    init_hermes_db()
    with connect() as c:
        cur = c.execute(
            """INSERT INTO hermes_traces
                   (user_id, query, embedding, intent, graph_mode,
                    num_chunks, fidelity_warnings, tokens, corrections_applied)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                user_id,
                query,
                _emb_to_blob(embedding),
                intent,
                graph_mode,
                int(num_chunks),
                int(fidelity_warnings),
                int(tokens),
                json.dumps(list(corrections_applied)),
            ),
        )
        return int(cur.lastrowid)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """Unit-normalise; safe on zero vectors."""
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else v / n


def nearest_traces(
    embedding: np.ndarray,
    *,
    user_id: int | None = None,
    k: int = 10,
    min_similarity: float = 0.45,
) -> list[Trace]:
    """Return up to k past traces most similar to `embedding` (cosine ≥ threshold).

    Scoped to user_id when provided so corrections learn from that user's own
    history. Without it, anonymous traces are searched.
    """
    init_hermes_db()
    with connect() as c:
        if user_id is None:
            rows = c.execute(
                "SELECT * FROM hermes_traces WHERE user_id IS NULL ORDER BY id DESC LIMIT 2000"
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM hermes_traces WHERE user_id = ? ORDER BY id DESC LIMIT 2000",
                (user_id,),
            ).fetchall()
    if not rows:
        return []
    q = _l2_normalize(np.asarray(embedding, dtype=np.float32))
    scored: list[Trace] = []
    for r in rows:
        t = _row_to_trace(r)
        e = _l2_normalize(t.embedding)
        if e.shape != q.shape:
            continue
        sim = float(np.dot(q, e))
        if sim >= min_similarity:
            t.similarity = sim
            scored.append(t)
    scored.sort(key=lambda x: x.similarity, reverse=True)
    return scored[:k]


def recent_traces(user_id: int | None = None, limit: int = 50) -> list[dict]:
    """Newest-first traces for inspection endpoints; no embeddings in the payload."""
    init_hermes_db()
    with connect() as c:
        if user_id is None:
            rows = c.execute(
                "SELECT id, user_id, query, intent, graph_mode, num_chunks, "
                "       fidelity_warnings, tokens, corrections_applied, created_at "
                "FROM hermes_traces ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT id, user_id, query, intent, graph_mode, num_chunks, "
                "       fidelity_warnings, tokens, corrections_applied, created_at "
                "FROM hermes_traces WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["corrections_applied"] = json.loads(d["corrections_applied"] or "[]")
        out.append(d)
    return out


def total_traces(user_id: int | None = None) -> int:
    """Count of traces (scoped if user_id given) — used by /hermes/stats."""
    init_hermes_db()
    with connect() as c:
        if user_id is None:
            return int(c.execute("SELECT COUNT(*) AS n FROM hermes_traces").fetchone()["n"])
        return int(
            c.execute(
                "SELECT COUNT(*) AS n FROM hermes_traces WHERE user_id = ?", (user_id,)
            ).fetchone()["n"]
        )
