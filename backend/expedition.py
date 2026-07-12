"""Expeditions — the narrative arc over the cognitive loop.

An expedition is a short chain of missions around one anchor node (a word
family, a sound set, a bridge cluster). It exists to make the planner's
differentiation FEEL like a journey instead of random cards: the child is
"exploring the UNI world", not answering item 47.

Design: the planner still owns node selection *between* expeditions (tier
rotation, ZPD); the expedition pins selection *within* an arc to the anchor's
own neighborhood, chosen at start. Completing all steps unlocks the world —
the payoff the universe view renders.

No LLM anywhere. SQLite, same db as the twin.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass

from evidence_graph import DB_PATH

STEPS_PER_EXPEDITION = 4


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS expeditions (
        student_id TEXT NOT NULL,
        anchor_id  TEXT NOT NULL,
        display    TEXT NOT NULL,
        tier       TEXT NOT NULL,
        node_queue TEXT NOT NULL,          -- JSON list of node_ids remaining
        steps_total INTEGER NOT NULL,
        steps_done  INTEGER NOT NULL DEFAULT 0,
        status     TEXT NOT NULL DEFAULT 'active',   -- active | complete
        started_ts REAL NOT NULL,
        completed_ts REAL
    )""")
    c.execute("CREATE INDEX IF NOT EXISTS ix_exp_student ON expeditions(student_id, status)")
    return c


@dataclass
class Expedition:
    anchor_id: str
    display: str
    tier: str
    queue: list[str]
    steps_total: int
    steps_done: int
    status: str


def get_active(student_id: str) -> Expedition | None:
    with _conn() as c:
        row = c.execute(
            "SELECT anchor_id, display, tier, node_queue, steps_total, steps_done, status "
            "FROM expeditions WHERE student_id=? AND status='active' "
            "ORDER BY started_ts DESC LIMIT 1", (student_id,)).fetchone()
    if not row:
        return None
    return Expedition(row[0], row[1], row[2], json.loads(row[3]), row[4], row[5], row[6])


def start(student_id: str, anchor_id: str, display: str, tier: str,
          node_queue: list[str]) -> Expedition:
    queue = node_queue[:STEPS_PER_EXPEDITION]
    with _conn() as c:
        c.execute("INSERT INTO expeditions VALUES (?,?,?,?,?,?,0,'active',?,NULL)",
                  (student_id, anchor_id, display, tier, json.dumps(queue),
                   len(queue), time.time()))
    return Expedition(anchor_id, display, tier, queue, len(queue), 0, "active")


def advance(student_id: str, node_id: str) -> Expedition | None:
    """Mark one step done when evidence arrives for the current queue head.
    Returns the updated expedition (status may flip to complete)."""
    exp = get_active(student_id)
    if exp is None or not exp.queue or exp.queue[0] != node_id:
        return exp
    exp.queue.pop(0)
    exp.steps_done += 1
    done = not exp.queue
    with _conn() as c:
        c.execute(
            "UPDATE expeditions SET node_queue=?, steps_done=?, status=?, completed_ts=? "
            "WHERE student_id=? AND anchor_id=? AND status='active'",
            (json.dumps(exp.queue), exp.steps_done,
             "complete" if done else "active",
             time.time() if done else None,
             student_id, exp.anchor_id))
    exp.status = "complete" if done else "active"
    return exp


def unlocked_worlds(student_id: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT anchor_id, display, tier, completed_ts FROM expeditions "
            "WHERE student_id=? AND status='complete' ORDER BY completed_ts",
            (student_id,)).fetchall()
    return [{"anchor_id": r[0], "display": r[1], "tier": r[2]} for r in rows]
