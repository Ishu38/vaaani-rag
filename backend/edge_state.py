"""Edge State — CASCADE edge-sequencing learner state.

Architecture position: the edge-level counterpart of the Cognitive Twin.
Every edge (u, v, type) in the linguistic graph has a belief tracked
independently. The child's language knowledge is modeled as the set
of acquired EDGES, not nodes — because language is relationships.

Provides: edge mastery, edge frontier computation, betweenness scoring,
and the edge-first query API the CASCADE planner needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import cognitive_twin as twin
from cognitive_twin import (
    PRIOR, MASTERED_AT, EdgeBelief, get_edge, snapshot_edges,
    update_edge, propagate_edge_evidence
)


def edge_mastery(student_id: str, edge_key: str) -> float:
    """Current mastery belief for this edge (decayed)."""
    return get_edge(student_id, edge_key).mastery


def is_edge_mastered(student_id: str, edge_key: str,
                     threshold: float = MASTERED_AT) -> bool:
    return get_edge(student_id, edge_key).mastery >= threshold


def edge_count(student_id: str) -> int:
    """How many edges has this learner encountered?"""
    return len(snapshot_edges(student_id))


def mastered_edges(student_id: str,
                   threshold: float = MASTERED_AT) -> list[str]:
    """Edge keys the learner has mastered."""
    return [k for k, b in snapshot_edges(student_id).items()
            if b.mastery >= threshold]


def learning_rate(student_id: str, window: int = 20) -> float:
    """Estimated learning rate — fraction of recent edges that were mastered
    within first few exposures."""
    edges = snapshot_edges(student_id)
    if not edges:
        return twin.TRANSIT
    mastered_in_window = sum(
        1 for b in edges.values()
        if b.mastered and b.exposures <= 3
    )
    return max(0.05, mastered_in_window / max(1, len(edges)))


def node_mastery_from_edges(student_id: str, node_id: str,
                            graph_edges: dict) -> float:
    """Compute effective node mastery from incident edge beliefs.

    A node is 'known' to the extent that its incident edges are learned.
    This is the CASCADE alternative to direct node BKT — mastery
    emerges from the edge subgraph.
    """
    incident_beliefs = []
    for etype, pairs in graph_edges.items():
        for s, t in pairs:
            if s == node_id or t == node_id:
                key = f"{s}::{t}::{etype}"
                incident_beliefs.append(get_edge(student_id, key).mastery)
    if not incident_beliefs:
        return twin.get(student_id, node_id).mastery  # fallback
    return sum(incident_beliefs) / len(incident_beliefs)


@dataclass
class EdgeFrontierCandidate:
    edge_key: str
    source: str
    target: str
    etype: str
    source_display: str
    target_display: str
    p_success: float
    mastery: float
    readiness: float
    curvature: float = 0.5         # Forman-Ricci (0=hard, 1=easy)
    structural_importance: float = 0.0  # betweenness in fringe


def edge_frontier_candidates(
    student_id: str,
    graph_edges: dict,
    node_display: callable,
    min_incident_mastery: float = 0.30,
    limit: int = 20,
) -> list[EdgeFrontierCandidate]:
    """Build CASCADE edge-level ZPD frontier.

    An edge is a candidate when at least one endpoint has a node
    belief above min_incident_mastery AND the edge itself is not
    yet mastered.

    Returns candidates sorted by structural importance (desc),
    then by p_success (closest to 0.70).
    """
    from ricci_curvature import get_curvature

    candidates = []
    for etype, pairs in graph_edges.items():
        for s, t in pairs:
            key = f"{s}::{t}::{etype}"
            bel = get_edge(student_id, key)

            # Already mastered — skip
            if bel.mastered:
                continue

            # At least one endpoint must be 'known'
            src_m = twin.get(student_id, s).mastery
            tgt_m = twin.get(student_id, t).mastery
            if src_m < min_incident_mastery and tgt_m < min_incident_mastery:
                continue

            # Compute p_success for this edge
            readiness = (src_m + tgt_m) / 2
            p = twin.GUESS + (1 - twin.GUESS - twin.SLIP) * (
                0.35 * bel.mastery + 0.65 * readiness)
            p = min(0.95, max(0.05, p))

            if p < 0.50 or p > 0.95:
                continue  # outside learnable range

            kappa = get_curvature(key)

            candidates.append(EdgeFrontierCandidate(
                edge_key=key, source=s, target=t, etype=etype,
                source_display=node_display(s),
                target_display=node_display(t),
                p_success=p, mastery=bel.mastery, readiness=readiness,
                curvature=kappa,
            ))

    if not candidates:
        return []

    # Structural importance = precomputed GLOBAL edge betweenness (stable bridge
    # signal; validated by research/curvature_sequencing.py — bridges beat easy
    # in-cluster edges for reaching percolation). Cheaper than per-call recompute.
    from betweenness import get_betweenness
    for c in candidates:
        c.structural_importance = get_betweenness(c.edge_key)

    # Sort: highest structural importance first, then closest to 0.70
    candidates.sort(key=lambda c: (
        -c.structural_importance,
        -abs(c.p_success - 0.70)
    ))
    return candidates[:limit]


def _score_betweenness(candidates: list[EdgeFrontierCandidate],
                       graph_edges: dict) -> None:
    """Approximate edge betweenness in the fringe subgraph.

    An edge with endpoints that appear in many OTHER candidate edges
    is structurally important — learning it connects many things.
    """
    node_freq: dict[str, int] = {}
    for c in candidates:
        node_freq[c.source] = node_freq.get(c.source, 0) + 1
        node_freq[c.target] = node_freq.get(c.target, 0) + 1
    max_freq = max(node_freq.values()) if node_freq else 1
    for c in candidates:
        c.structural_importance = round(
            (node_freq.get(c.source, 0) + node_freq.get(c.target, 0))
            / (2 * max_freq + 1e-12), 3)
