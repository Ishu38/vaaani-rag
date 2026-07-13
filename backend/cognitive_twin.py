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

CASCADE (2026-07-13): edge-level BKT. Every graph edge (u, v, type) also
has a twin belief — the child's understanding of that linguistic relationship.
The same BKT math applies; evidence on an edge updates it independently.
Evidence propagation: when one edge is mastered, adjacent edges receive a
damped boost (the percolation effect).
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

# CASCADE: evidence propagation parameters
EDGE_PROPAGATION_STRENGTH = 0.30   # δ — how much evidence spills to neighbors
EDGE_PROPAGATION_DECAY = 0.50      # η — how fast spill decays with distance
EDGE_PROPAGATION_HOPS = 2          # h — maximum hop distance for propagation


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
    # CASCADE edge-level belief table
    c.execute("""CREATE TABLE IF NOT EXISTS twin_edge (
        student_id TEXT NOT NULL,
        edge_key   TEXT NOT NULL,
        source     TEXT NOT NULL,
        target     TEXT NOT NULL,
        etype      TEXT NOT NULL,
        mastery    REAL NOT NULL,
        exposures  INTEGER NOT NULL DEFAULT 0,
        updated_ts REAL NOT NULL,
        PRIMARY KEY (student_id, edge_key)
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


@dataclass
class EdgeBelief:
    """CASCADE: BKT belief over one linguistic relationship (edge)."""
    edge_key: str
    source: str
    target: str
    etype: str
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


# ── CASCADE: Edge-level BKT ──────────────────────────────────────────

def get_edge(student_id: str, edge_key: str) -> EdgeBelief:
    with _conn() as c:
        row = c.execute(
            "SELECT source, target, etype, mastery, exposures, updated_ts "
            "FROM twin_edge WHERE student_id=? AND edge_key=?",
            (student_id, edge_key)).fetchone()
    if row is None:
        src, tgt, et = _parse_edge_key(edge_key)
        return EdgeBelief(edge_key, src, tgt, et, PRIOR, 0, time.time())
    src, tgt, et, m, ex, ts = row
    return EdgeBelief(edge_key, src, tgt, et, _decayed(m, ts), ex, ts)


def snapshot_edges(student_id: str) -> dict[str, EdgeBelief]:
    with _conn() as c:
        rows = c.execute(
            "SELECT edge_key, source, target, etype, mastery, exposures, updated_ts "
            "FROM twin_edge WHERE student_id=?", (student_id,)).fetchall()
    return {k: EdgeBelief(k, s, t, e, _decayed(m, ts), ex, ts)
            for k, s, t, e, m, ex, ts in rows}


def _parse_edge_key(key: str) -> tuple[str, str, str]:
    """edge_key = 'source::target::type'"""
    parts = key.split("::")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return key, key, "related_to"


def update_edge(ev: EvidenceObject) -> EdgeBelief:
    """Update the BKT belief for an edge from evidence."""
    edge_key = ev.edge_key
    if not edge_key:
        src, tgt, et = _parse_edge_key("::")
        return EdgeBelief("", "", "", "", PRIOR, 0, time.time())

    b = get_edge(ev.student_id, edge_key)
    p = b.mastery

    w = ev.confidence
    outcome_correct = ev.outcome == "correct"
    if ev.outcome == "partial":
        outcome_correct, w = True, w * 0.5

    if outcome_correct:
        post = p * (1 - SLIP) / (p * (1 - SLIP) + (1 - p) * GUESS + 1e-12)
    else:
        post = p * SLIP / (p * SLIP + (1 - p) * (1 - SLIP) + 1e-12)
    post = (1 - w) * p + w * post
    post = post + (1 - post) * TRANSIT

    src, tgt, et = _parse_edge_key(edge_key)
    now = time.time()
    with _conn() as c:
        c.execute("INSERT INTO twin_edge VALUES (?,?,?,?,?,?,?,?) "
                  "ON CONFLICT(student_id, edge_key) DO UPDATE SET "
                  "mastery=excluded.mastery, exposures=twin_edge.exposures+1, "
                  "updated_ts=excluded.updated_ts",
                  (ev.student_id, edge_key, src, tgt, et, post, 1, now))
    return EdgeBelief(edge_key, src, tgt, et, post, b.exposures + 1, now)


def propagate_edge_evidence(ev: EvidenceObject, graph_edges: dict) -> None:
    """CASCADE evidence propagation: mastering one edge boosts neighbors.

    For each edge within h hops of ev.edge_key, apply a damped update.
    δ = EDGE_PROPAGATION_STRENGTH, η = EDGE_PROPAGATION_DECAY,
    hop distance exponentially reduces the effect.
    """
    edge_key = ev.edge_key
    if not edge_key or not graph_edges:
        return

    outcome_correct = ev.outcome == "correct"
    spoke = _edge_neighborhood(edge_key, graph_edges,
                               max_hops=EDGE_PROPAGATION_HOPS)
    for neighbor_key, hop_distance in spoke.items():
        if neighbor_key == edge_key:
            continue
        delta = (EDGE_PROPAGATION_STRENGTH *
                 (EDGE_PROPAGATION_DECAY ** hop_distance))
        b = get_edge(ev.student_id, neighbor_key)
        old = b.mastery
        if outcome_correct:
            new = old + (1 - old) * delta * 0.15
        else:
            new = old - old * delta * 0.08
            new = max(PRIOR, new)
        if abs(new - old) < 0.001:
            continue
        src, tgt, et = _parse_edge_key(neighbor_key)
        now = time.time()
        with _conn() as c:
            c.execute("INSERT INTO twin_edge VALUES (?,?,?,?,?,?,?,?) "
                      "ON CONFLICT(student_id, edge_key) DO UPDATE SET "
                      "mastery=excluded.mastery, "
                      "exposures=twin_edge.exposures,"
                      "updated_ts=excluded.updated_ts",
                      (ev.student_id, neighbor_key, src, tgt, et, new,
                       b.exposures, now))


def _edge_neighborhood(edge_key: str, graph_edges: dict,
                       max_hops: int = 2) -> dict[str, int]:
    """BFS from an edge through the graph — returns {edge_key: hop_distance}."""
    src, tgt, _ = _parse_edge_key(edge_key)
    visited: dict[str, int] = {edge_key: 0}
    frontier = [src, tgt]
    for hop in range(1, max_hops + 1):
        next_frontier = []
        for node in frontier:
            for etype, pairs in graph_edges.items():
                for s, t in pairs:
                    adj_key = f"{s}::{t}::{etype}"
                    if adj_key in visited:
                        continue
                    if s == node or t == node:
                        visited[adj_key] = hop
                        next_frontier.extend([s, t])
        frontier = list(set(next_frontier) - set(visited.keys()))
    return visited


def mastered_nodes_percolation(student_id: str, theta: float = 0.90) -> set[str]:
    """CASCADE: nodes mastered via edge-percolation threshold.

    A node is percolated (mastered) if enough incident edges are learned
    that the percolation probability exceeds the threshold. This is the
    emergent mastery definition — you don't know "run" in isolation,
    you know it when "runs", "running", and "ran" are also learned.
    """
    from development_engine import get_world_edges
    from percolation import percolated_nodes
    return percolated_nodes(
        snapshot_edges(student_id),
        get_world_edges(),
        theta=theta, k_min=3,
    )


def update(ev: EvidenceObject) -> NodeBelief:
    """Ingest one evidence object: persist it, run the BKT update, return belief.

    Perceptual confidence w blends the posterior with the prior belief:
    w=1 is a fully trusted observation; w=0 changes nothing (AIMA soft evidence).
    'partial' outcomes count as a half-weight correct.

    CASCADE: when ev.edge_key is set, also updates the edge-level belief
    and propagates evidence to adjacent edges.
    """
    record_evidence(ev)
    import credal
    credal.update(ev)

    # ── Node-level update (always) ────────────────────────────
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
    post = (1 - w) * p + w * post
    post = post + (1 - post) * TRANSIT

    now = time.time()
    with _conn() as c:
        c.execute("INSERT INTO twin VALUES (?,?,?,?,?) "
                  "ON CONFLICT(student_id, node_id) DO UPDATE SET "
                  "mastery=excluded.mastery, exposures=twin.exposures+1, "
                  "updated_ts=excluded.updated_ts",
                  (ev.student_id, ev.node_id, post, 1, now))
        c.execute("UPDATE predictions SET actual=? WHERE student_id=? AND "
                  "node_id=? AND actual IS NULL",
                  (ev.outcome, ev.student_id, ev.node_id))

    # ── CASCADE: Edge-level update + propagation ──────────────
    if ev.edge_key:
        try:
            update_edge(ev)
            # Evidence percolation: mastering one edge strengthens nearby edges
            from development_engine import get_world_edges
            propagate_edge_evidence(ev, get_world_edges())
        except Exception:
            pass

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
