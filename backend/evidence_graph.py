"""Evidence Object Graph — the typed record every perception emits.

Architecture position (Neil's diagram, 2026-07-12):

    PERCEPTION -> **Evidence Object Graph** -> Neuro-Symbolic Reasoner -> ...

Every perceptual subsystem (quiz answer, mission completion, spaced review,
chat interaction, OCR'd worksheet, audio attempt) reduces its observation to
one EvidenceObject linked to a node of the language graph. Nothing else in
the architecture ever consumes raw input — only evidence. AIMA framing: the
percept-to-evidence boundary of a hybrid agent (ch. 2); each record is one
observation for the Cognitive Twin's filtering update (ch. 14).

Storage: SQLite (data/cognitive_twin.db, shared with the twin). The "graph"
in the name is the link structure: evidence -> graph node -> other evidence,
queryable per node and per learner.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    from config import DATA_DIR
except ImportError:  # standalone use
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DB_PATH = Path(DATA_DIR) / "cognitive_twin.db"

SOURCES = ("quiz", "mission", "review", "chat", "ocr", "audio", "seed")
OUTCOMES = ("correct", "incorrect", "partial")


@dataclass
class EvidenceObject:
    student_id: str
    node_id: str              # id in data/graph.json
    source: str               # one of SOURCES
    outcome: str              # one of OUTCOMES
    confidence: float = 1.0   # perceptual confidence in the observation itself
    ts: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)
    evidence_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self):
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        if self.outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {self.outcome!r}")
        self.confidence = float(min(1.0, max(0.0, self.confidence)))


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS evidence (
        evidence_id TEXT PRIMARY KEY,
        student_id  TEXT NOT NULL,
        node_id     TEXT NOT NULL,
        source      TEXT NOT NULL,
        outcome     TEXT NOT NULL,
        confidence  REAL NOT NULL,
        ts          REAL NOT NULL,
        meta        TEXT NOT NULL DEFAULT '{}'
    )""")
    c.execute("CREATE INDEX IF NOT EXISTS ix_ev_student_node ON evidence(student_id, node_id)")
    return c


def record(ev: EvidenceObject) -> str:
    """Persist one evidence object; returns its id."""
    with _conn() as c:
        c.execute(
            "INSERT INTO evidence VALUES (?,?,?,?,?,?,?,?)",
            (ev.evidence_id, ev.student_id, ev.node_id, ev.source,
             ev.outcome, ev.confidence, ev.ts, json.dumps(ev.meta)),
        )
    return ev.evidence_id


def for_node(student_id: str, node_id: str, limit: int = 50) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM evidence WHERE student_id=? AND node_id=? "
            "ORDER BY ts DESC LIMIT ?", (student_id, node_id, limit)).fetchall()
    cols = ["evidence_id", "student_id", "node_id", "source", "outcome",
            "confidence", "ts", "meta"]
    return [dict(zip(cols, r)) for r in rows]


def count(student_id: str) -> int:
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM evidence WHERE student_id=?",
                         (student_id,)).fetchone()[0]


def recent_nodes(student_id: str, source: str | None = None, limit: int = 8) -> list[str]:
    """Most-recent node_ids for a student (newest first), optionally by source.
    Used by the planner's variety pressure — what did we just do?"""
    q = "SELECT node_id FROM evidence WHERE student_id=?"
    args: list = [student_id]
    if source:
        q += " AND source=?"
        args.append(source)
    q += " ORDER BY ts DESC LIMIT ?"
    args.append(limit)
    with _conn() as c:
        return [r[0] for r in c.execute(q, args).fetchall()]
