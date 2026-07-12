"""Credal belief channel — per-node Beta(α, β) uncertainty (theory O2).

Position (LINGUISTIC_STATE_THEORY.md §1.2): U_t, the quantified-uncertainty
component of Λ_t. This is a SECOND derived view of the same evidence log the
Cognitive Twin folds — it does not replace BKT mastery. The two channels
measure different things:

    twin.mastery   P(node known)          — latent, model-based (BKT)
    credal μ       P(observed correct)    — observable accuracy, Beta-tracked

The model links them: μ ≈ γ + (1−γ−σ)·mastery, so inverting the Beta mean
gives an independent mastery estimate — `misfit()` reports the disagreement,
a built-in model-criticism diagnostic per node.

Update law (Jeffrey-weighted conjugate step, weight w = evidence confidence;
'partial' counts as correct at half weight, mirroring the twin):

    correct:    α += w          incorrect:  β += w

Decay (uncertainty grows with disuse — credal analogue of twin forgetting):
counts relax toward the prior pseudo-counts with time constant TAU_U:

    α ← A0 + (α − A0)·exp(−dt/TAU_U)      (same for β toward B0)

Proved properties (verified numerically over unit + fractional weights,
2026-07-12; proof in COGNITIVE_STATE_ALGEBRA.md T4/T5):

    T4  mean-confirming evidence never increases Var(θ)
        (success when μ ≥ ½, failure when μ ≤ ½)
    T5  EIG(one more observation) = μ(1−μ)/(α+β+1)²    (law of total variance)

Storage: `credal` table in the same SQLite DB as the twin. Replayable from
the evidence log at any time (`replay()`) — law A5 in action.
"""

from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass

from evidence_graph import DB_PATH, EvidenceObject

# Prior: pseudo-counts matching the BKT prior's expected observable accuracy
# θ0 = γ + (1−γ−σ)·π = 0.27, at weak strength N0 = 2 effective observations.
from cognitive_twin import GUESS, SLIP, PRIOR, TAU_DAYS

THETA0 = GUESS + (1.0 - GUESS - SLIP) * PRIOR      # 0.27 with defaults
N0 = 2.0
A0, B0 = THETA0 * N0, (1.0 - THETA0) * N0
TAU_U = TAU_DAYS                                    # separately fittable (H7)


@dataclass
class CredalBelief:
    node_id: str
    alpha: float
    beta: float
    updated_ts: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def n_eff(self) -> float:
        """Effective evidence mass behind this belief."""
        return self.alpha + self.beta

    @property
    def variance(self) -> float:
        s = self.alpha + self.beta
        return self.alpha * self.beta / (s * s * (s + 1.0))

    @property
    def sd(self) -> float:
        return math.sqrt(self.variance)

    @property
    def eig(self) -> float:
        """Expected variance reduction from ONE more observation (T5)."""
        mu, s = self.mean, self.alpha + self.beta
        return mu * (1.0 - mu) / ((s + 1.0) ** 2)

    def misfit(self, mastery: float) -> float:
        """Model criticism: |Beta-implied mastery − BKT mastery|.
        Inverts μ = γ + (1−γ−σ)·p; clamped to [0,1]."""
        implied = (self.mean - GUESS) / (1.0 - GUESS - SLIP)
        implied = min(1.0, max(0.0, implied))
        return abs(implied - mastery)


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS credal (
        student_id TEXT NOT NULL,
        node_id    TEXT NOT NULL,
        alpha      REAL NOT NULL,
        beta       REAL NOT NULL,
        updated_ts REAL NOT NULL,
        PRIMARY KEY (student_id, node_id)
    )""")
    return c


def _decayed(alpha: float, beta: float, updated_ts: float,
             now: float | None = None) -> tuple[float, float]:
    now = now or time.time()
    dt_days = max(0.0, (now - updated_ts) / 86400.0)
    k = math.exp(-dt_days / TAU_U)
    return A0 + (alpha - A0) * k, B0 + (beta - B0) * k


def get(student_id: str, node_id: str) -> CredalBelief:
    with _conn() as c:
        row = c.execute("SELECT alpha, beta, updated_ts FROM credal "
                        "WHERE student_id=? AND node_id=?",
                        (student_id, node_id)).fetchone()
    if row is None:
        return CredalBelief(node_id, A0, B0, time.time())
    a, b, ts = row
    a, b = _decayed(a, b, ts)
    return CredalBelief(node_id, a, b, ts)


def snapshot(student_id: str) -> dict[str, CredalBelief]:
    with _conn() as c:
        rows = c.execute("SELECT node_id, alpha, beta, updated_ts FROM credal "
                         "WHERE student_id=?", (student_id,)).fetchall()
    out = {}
    for n, a, b, ts in rows:
        a, b = _decayed(a, b, ts)
        out[n] = CredalBelief(n, a, b, ts)
    return out


def update(ev: EvidenceObject, now: float | None = None) -> CredalBelief:
    """Fold one evidence object into the credal channel. Called by
    twin.update() so the two views share the single evidence ingress (A1)."""
    now = now or time.time()
    cur = get(ev.student_id, ev.node_id)
    a, b = _decayed(cur.alpha, cur.beta, cur.updated_ts, now)

    w = ev.confidence
    outcome_correct = ev.outcome == "correct"
    if ev.outcome == "partial":
        outcome_correct, w = True, w * 0.5
    if outcome_correct:
        a += w
    else:
        b += w

    with _conn() as c:
        c.execute("INSERT INTO credal VALUES (?,?,?,?,?) "
                  "ON CONFLICT(student_id, node_id) DO UPDATE SET "
                  "alpha=excluded.alpha, beta=excluded.beta, "
                  "updated_ts=excluded.updated_ts",
                  (ev.student_id, ev.node_id, a, b, now))
    return CredalBelief(ev.node_id, a, b, now)


def replay(student_id: str | None = None) -> int:
    """Rebuild the credal table from the evidence log (replayability, A5).
    Used to backfill histories recorded before this channel existed.
    Returns the number of evidence objects folded."""
    q = "SELECT student_id, node_id, outcome, confidence, ts FROM evidence"
    args: tuple = ()
    if student_id:
        q += " WHERE student_id=?"
        args = (student_id,)
    q += " ORDER BY ts, evidence_id"
    with _conn() as c:
        rows = c.execute(q, args).fetchall()
        if student_id:
            c.execute("DELETE FROM credal WHERE student_id=?", (student_id,))
        else:
            c.execute("DELETE FROM credal")

    state: dict[tuple[str, str], tuple[float, float, float]] = {}
    for sid, nid, outcome, conf, ts in rows:
        a, b, prev_ts = state.get((sid, nid), (A0, B0, ts))
        a, b = _decayed(a, b, prev_ts, ts)
        w = conf
        correct = outcome == "correct"
        if outcome == "partial":
            correct, w = True, w * 0.5
        if correct:
            a += w
        else:
            b += w
        state[(sid, nid)] = (a, b, ts)

    with _conn() as c:
        c.executemany(
            "INSERT OR REPLACE INTO credal VALUES (?,?,?,?,?)",
            [(sid, nid, a, b, ts) for (sid, nid), (a, b, ts) in state.items()])
    return len(rows)
