"""Development Engine — predicts the learner's next stage of growth.

Architecture position (Neil's diagram, 2026-07-12): consumes the Linguistic
World Model (data/graph.json — 785 nodes, `depends_on` edges) and the
Cognitive Twin; produces the ZPD FRONTIER: the ordered set of nodes the
learner is ready to grow into next.

Formalism: prediction over the belief state (AIMA ch. 14). The Zone of
Proximal Development is made numeric: a node is on the frontier when its
predicted P(success) falls in [ZPD_LO, ZPD_HI] given the twin's mastery of
the node and its prerequisites.

    p_success = GUESS + (1 - GUESS - SLIP) * (0.35*mastery + 0.65*readiness)
    readiness = mean mastery of depends_on prerequisites (1.0 if none)

Below ZPD_LO: frustration zone — the developmental firewall's territory.
Above ZPD_HI: comfort zone — no growth. The v0 blend is a documented
heuristic; the Metacognitive Evaluation stage's calibration table is what
will tune it against reality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cognitive_twin as twin
from cognitive_twin import GUESS, SLIP, MASTERED_AT

try:
    from config import GRAPH_PATH
except ImportError:
    GRAPH_PATH = Path(__file__).resolve().parent.parent / "data" / "graph.json"

ZPD_LO, ZPD_HI = 0.60, 0.80


@dataclass
class FrontierNode:
    node_id: str
    display: str
    p_success: float
    mastery: float
    readiness: float
    prerequisites: list[str]


class WorldModel:
    """Read-only view of the language graph (the Linguistic World Model store)."""

    def __init__(self, path: Path | str = GRAPH_PATH):
        g = json.loads(Path(path).read_text())
        self.nodes: dict[str, dict] = {n["id"]: n for n in g["nodes"]}
        self.prereqs: dict[str, list[str]] = {}
        for e in g["links"]:
            t = e.get("type")
            if t == "depends_on":
                # A depends_on B  =>  B is prerequisite of A
                self.prereqs.setdefault(e["source"], []).append(e["target"])
            elif t == "prerequisite_for":
                # A prerequisite_for B  =>  A is prerequisite of B
                self.prereqs.setdefault(e["target"], []).append(e["source"])

        # Derived-prerequisite overlay (scripts/enrich_prereqs.py) — additive,
        # revertible by deleting the file; source graph is never mutated.
        overlay = Path(path).parent / "prereq_overlay.json"
        if overlay.exists():
            for d in json.loads(overlay.read_text()).get("edges", []):
                pres = self.prereqs.setdefault(d["for"], [])
                if d["prerequisite"] not in pres:
                    pres.append(d["prerequisite"])

    def display(self, node_id: str) -> str:
        return self.nodes.get(node_id, {}).get("display", node_id)


# CASCADE: the WorldModel also holds the edge graph
_WORLD_EDGES: dict[str, list[tuple[str, str]]] | None = None


def get_world_edges(world: WorldModel | None = None) -> dict[str, list[tuple[str, str]]]:
    """Lazy-load edge index from the graph. Shared across all callers."""
    global _WORLD_EDGES
    if _WORLD_EDGES is None:
        g = json.loads(Path(GRAPH_PATH).read_text())
        edges: dict[str, list[tuple[str, str]]] = {}
        for e in g.get("links", []):
            t = e.get("type", "?")
            edges.setdefault(t, []).append((e["source"], e["target"]))
        _WORLD_EDGES = edges
    return _WORLD_EDGES


def p_success(student_id: str, node_id: str, world: WorldModel) -> tuple[float, float, float]:
    """Return (p_success, mastery, readiness) for one node.

    Uses dynamically-tuned blend weights from calibration data.
    Defaults to 0.35/0.65 until calibration data drives a tune."""
    b = twin.get(student_id, node_id)
    prereqs = world.prereqs.get(node_id, [])
    if prereqs:
        readiness = sum(twin.get(student_id, p).mastery for p in prereqs) / len(prereqs)
    else:
        readiness = 1.0
    w_m, w_r = get_blend_weights()
    p = GUESS + (1 - GUESS - SLIP) * (w_m * b.mastery + w_r * readiness)
    return p, b.mastery, readiness


def frontier(student_id: str, world: WorldModel | None = None,
             limit: int = 20) -> list[FrontierNode]:
    """The learner's predicted next stage: ZPD-band nodes, best-first."""
    world = world or WorldModel()
    out: list[FrontierNode] = []
    for node_id in world.nodes:
        p, m, r = p_success(student_id, node_id, world)
        if m >= MASTERED_AT or not (ZPD_LO <= p <= ZPD_HI):
            continue
        out.append(FrontierNode(node_id, world.display(node_id), p, m, r,
                                world.prereqs.get(node_id, [])))
    # closest to the sweet spot first; among ties prefer nodes whose
    # prerequisite structure is actually known (structure-informed > generic)
    mid = (ZPD_LO + ZPD_HI) / 2
    out.sort(key=lambda f: (round(abs(f.p_success - mid), 3), -len(f.prerequisites)))
    return out[:limit]


# ── CASCADE: Edge-level Frontier ────────────────────────────────────

@dataclass
class EdgeFrontierNode:
    edge_key: str
    source: str
    target: str
    etype: str
    source_display: str
    target_display: str
    p_success: float
    mastery: float
    readiness: float
    source_mastery: float
    target_mastery: float


def p_success_edge(student_id: str, edge_key: str,
                   world: WorldModel) -> tuple[float, float, float, float, float]:
    """Return (p_success, edge_mastery, readiness, src_mastery, tgt_mastery).

    Readiness = mean mastery of the two endpoint nodes.
    Edge mastery from twin_edge BKT.
    """
    src, tgt, et = edge_key.split("::")
    b = twin.get_edge(student_id, edge_key)
    em = b.mastery
    sm = twin.get(student_id, src).mastery
    tm = twin.get(student_id, tgt).mastery
    readiness = (sm + tm) / 2
    w_m, w_r = get_blend_weights()
    p = GUESS + (1 - GUESS - SLIP) * (w_m * em + w_r * readiness)
    return p, em, readiness, sm, tm


def edge_frontier(student_id: str, world: WorldModel | None = None,
                  limit: int = 20) -> list[EdgeFrontierNode]:
    """CASCADE: edge-level ZPD frontier.

    An edge is a candidate when at least one endpoint has node mastery
    above 0.20 AND the edge p_success falls in [0.50, 0.95].
    Sorted by p_success closest to 0.70.
    """
    world = world or WorldModel()
    edges = get_world_edges()
    out: list[EdgeFrontierNode] = []
    for etype, pairs in edges.items():
        for s, t in pairs:
            key = f"{s}::{t}::{etype}"
            p, em, r, sm, tm = p_success_edge(student_id, key, world)
            if sm < 0.20 and tm < 0.20:
                continue
            if not (0.50 <= p <= 0.95):
                continue
            if twin.get_edge(student_id, key).mastered:
                continue
            out.append(EdgeFrontierNode(
                key, s, t, etype,
                world.display(s), world.display(t),
                p, em, r, sm, tm))
    mid = 0.70
    out.sort(key=lambda f: abs(f.p_success - mid))
    return out[:limit]


# ── Calibration-Driven Parameter Tuning ─────────────────────────────

# The mastery/readiness blend weight in p_success().  Defaults are a
# documented heuristic; the Metacognitive Evaluation stage tunes them
# against reality by reading the twin's calibration table.

_BLEND_MASTERY = 0.35
_BLEND_READINESS = 0.65


def get_blend_weights() -> tuple[float, float]:
    """Return the current (mastery_weight, readiness_weight) blend.

    These are dynamically tuned from the twin's calibration table so the
    system's predicted probabilities stay honest — if the twin systematically
    overestimates, readiness gets more weight (since readiness depends on
    prerequisites, which are more conservative). If it underestimates, mastery
    gets more weight.
    """
    return (_BLEND_MASTERY, _BLEND_READINESS)


def tune_blend_from_calibration(student_id: str | None = None) -> dict:
    """Read the twin's calibration table and adjust the blend weights.

    Called periodically (or on demand) after enough evidence accumulates.
    The rule: if actual success rate in the ZPD-targeted bins is
    significantly below predicted, the readiness weight is increased
    (conservatism); if significantly above, mastery gets more weight.

    Returns the tuning report for auditability.
    """
    global _BLEND_MASTERY, _BLEND_READINESS
    import cognitive_twin as twin

    cal = twin.calibration(student_id, bins=5)
    if not cal:
        return {"tuned": False, "reason": "no calibration data yet"}

    # Focus on the ZPD-relevant bins (0.60–0.80 is where the planner picks)
    zpd_bins = [b for b in cal
                if 0.50 <= float(b["bin"].split("-")[0]) <= 0.90]
    if not zpd_bins:
        return {"tuned": False, "reason": "no ZPD-range calibration data"}

    # Weighted average of the calibration error
    total_n = sum(b["n"] for b in zpd_bins)
    if total_n < 10:
        return {"tuned": False, "reason": f"only {total_n} observations — need ≥10"}

    weighted_error = sum(
        b["n"] * (b["predicted_mean"] - b["actual_rate"])
        for b in zpd_bins
    ) / total_n

    # Tuning: error > 0 means overprediction (the twin is too optimistic)
    # → increase readiness weight (more conservative)
    # error < 0 means underprediction → increase mastery weight
    adjustment = min(0.15, max(-0.15, weighted_error * 0.5))
    _BLEND_MASTERY = round(min(0.55, max(0.15, _BLEND_MASTERY - adjustment)), 3)
    _BLEND_READINESS = round(1.0 - _BLEND_MASTERY, 3)

    return {
        "tuned": True,
        "observations": total_n,
        "calibration_error": round(weighted_error, 4),
        "adjustment": round(adjustment, 4),
        "new_mastery_weight": _BLEND_MASTERY,
        "new_readiness_weight": _BLEND_READINESS,
    }
