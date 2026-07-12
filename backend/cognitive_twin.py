"""Cognitive Twin Engine — the continuously evolving learner belief state.

Architecture position (Neil's diagram, 2026-07-12): center of the loop.
Every EvidenceObject updates it; the Development Engine and Pedagogical
Planner read it; Metacognitive Evaluation audits it.

Formalism: Bayesian Knowledge Tracing — a two-state HMM per language-graph
node (AIMA ch. 14 filtering). Per (student, node):

    mastery  P(L)   — belief the learner knows the node
    slip     P(err | known)      default 0.10
    guess    P(correct | unknown) default 0.20
    transit  P(learning per exposure) default 0.15

Update on evidence (weighted by perceptual confidence), then a learning step.
Forgetting: mastery decays toward the prior with half-life TAU_DAYS between
observations, so spaced-review urgency falls out of the same state.

Metacognition hooks: every prediction the twin makes (P(success) issued to
the planner) is logged; when the outcome arrives, (predicted, actual) pairs
accumulate — calibration() reports how honest the twin's probabilities are.
That table is the Metacognitive Evaluation stage's raw material.
"""

from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from evidence_graph import DB_PATH, EvidenceObject, record as record_evidence

PRIOR = 0.10
SLIP = 0.10
GUESS = 0.20
TRANSIT = 0.15
TAU_DAYS = 14.0          # forgetting half-life-ish constant
MASTERED_AT = 0.95


def _conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS twin (
        student_id TEXT NOT NULL,
        node_id    TEXT NOT NULL,
        mastery    REAL NOT NULL,
        exposures  INTEGER NOT NULL DEFAULT 0,
        updated_ts REAL NOT NULL,
        PRIMARY KEY (student_id, node_id)
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS predictions (
        student_id TEXT NOT NULL,
        node_id    TEXT NOT NULL,
        predicted  REAL NOT NULL,
        ts         REAL NOT NULL,
        actual     TEXT
    )""")
    return c


@dataclass
class NodeBelief:
    node_id: str
    mastery: float
    exposures: int
    updated_ts: float

    @property
    def mastered(self) -> bool:
        return self.mastery >= MASTERED_AT


def _decayed(mastery: float, updated_ts: float, now: float | None = None) -> float:
    """Forgetting: decay toward PRIOR with time constant TAU_DAYS."""
    now = now or time.time()
    dt_days = max(0.0, (now - updated_ts) / 86400.0)
    k = math.exp(-dt_days / TAU_DAYS)
    return PRIOR + (mastery - PRIOR) * k


def get(student_id: str, node_id: str) -> NodeBelief:
    with _conn() as c:
        row = c.execute("SELECT mastery, exposures, updated_ts FROM twin "
                        "WHERE student_id=? AND node_id=?",
                        (student_id, node_id)).fetchone()
    if row is None:
        return NodeBelief(node_id, PRIOR, 0, time.time())
    m, ex, ts = row
    return NodeBelief(node_id, _decayed(m, ts), ex, ts)


def snapshot(student_id: str) -> dict[str, NodeBelief]:
    with _conn() as c:
        rows = c.execute("SELECT node_id, mastery, exposures, updated_ts "
                         "FROM twin WHERE student_id=?", (student_id,)).fetchall()
    return {n: NodeBelief(n, _decayed(m, ts), ex, ts) for n, m, ex, ts in rows}


def update(ev: EvidenceObject) -> NodeBelief:
    """Ingest one evidence object: persist it, run the BKT update, return belief.

    Perceptual confidence w blends the posterior with the prior belief:
    w=1 is a fully trusted observation; w=0 changes nothing (AIMA soft evidence).
    'partial' outcomes count as a half-weight correct.
    """
    record_evidence(ev)
    # Credal channel (theory O2): the same evidence also updates the per-node
    # Beta uncertainty view. Kept behind the twin's single ingress so law A1
    # (state changes only through evidence) covers both views.
    import credal
    credal.update(ev)
    b = get(ev.student_id, ev.node_id)
    p = b.mastery

    w = ev.confidence
    outcome_correct = ev.outcome == "correct"
    if ev.outcome == "partial":
        outcome_correct, w = True, w * 0.5

    if outcome_correct:
        post = p * (1 - SLIP) / (p * (1 - SLIP) + (1 - p) * GUESS + 1e-12)
    else:
        post = p * SLIP / (p * SLIP + (1 - p) * (1 - SLIP) + 1e-12)
    post = (1 - w) * p + w * post              # soft-evidence blend
    post = post + (1 - post) * TRANSIT         # learning step

    now = time.time()
    with _conn() as c:
        c.execute("INSERT INTO twin VALUES (?,?,?,?,?) "
                  "ON CONFLICT(student_id, node_id) DO UPDATE SET "
                  "mastery=excluded.mastery, exposures=twin.exposures+1, "
                  "updated_ts=excluded.updated_ts",
                  (ev.student_id, ev.node_id, post, 1, now))
        # metacognition: close any open prediction for this node
        c.execute("UPDATE predictions SET actual=? WHERE student_id=? AND "
                  "node_id=? AND actual IS NULL",
                  (ev.outcome, ev.student_id, ev.node_id))
    return NodeBelief(ev.node_id, post, b.exposures + 1, now)


def log_prediction(student_id: str, node_id: str, predicted: float) -> None:
    """Planner calls this when it assigns an activity with predicted P(success)."""
    with _conn() as c:
        c.execute("INSERT INTO predictions VALUES (?,?,?,?,NULL)",
                  (student_id, node_id, predicted, time.time()))


def calibration(student_id: str | None = None, bins: int = 5) -> list[dict]:
    """Metacognitive Evaluation: predicted-vs-actual reliability table."""
    q = "SELECT predicted, actual FROM predictions WHERE actual IS NOT NULL"
    args: tuple = ()
    if student_id:
        q += " AND student_id=?"
        args = (student_id,)
    with _conn() as c:
        rows = c.execute(q, args).fetchall()
    table = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        seg = [(p, a) for p, a in rows if lo <= p < hi or (i == bins - 1 and p == 1.0)]
        if not seg:
            continue
        actual_rate = sum(1 for _, a in seg if a in ("correct", "partial")) / len(seg)
        table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(seg),
                      "predicted_mean": sum(p for p, _ in seg) / len(seg),
                      "actual_rate": actual_rate})
    return table
