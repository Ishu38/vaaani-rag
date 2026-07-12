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


def p_success(student_id: str, node_id: str, world: WorldModel) -> tuple[float, float, float]:
    """Return (p_success, mastery, readiness) for one node."""
    b = twin.get(student_id, node_id)
    prereqs = world.prereqs.get(node_id, [])
    if prereqs:
        readiness = sum(twin.get(student_id, p).mastery for p in prereqs) / len(prereqs)
    else:
        readiness = 1.0
    p = GUESS + (1 - GUESS - SLIP) * (0.35 * b.mastery + 0.65 * readiness)
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
