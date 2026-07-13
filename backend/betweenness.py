"""Edge betweenness — CASCADE bridge signal for curvature-guided sequencing.

Percolation reaches whole-graph mastery fastest when the learner is sequenced
through the BRIDGE edges that merge communities, not the easy in-cluster ones
(validated: research/curvature_sequencing.py — readiness-gated bridge-first
reaches 53% percolation vs easy-first's 33%). Betweenness centrality is the
gold-standard bridge signal, and it is a STATIC property of the graph, so we
precompute it once and cache — same pattern as ricci_curvature.

get_betweenness(edge_key) → [0,1], where 1 = strongest bridge in the graph.
The Pedagogical Planner reads this (readiness-gated) to prefer learnable bridges.
"""

from __future__ import annotations

_BETWEENNESS_CACHE: dict[str, float] | None = None


def _compute_all() -> dict[str, float]:
    import networkx as nx
    from development_engine import get_world_edges

    edges = get_world_edges()                      # {etype: [(s,t),...]}
    g = nx.Graph()
    key_pairs = []
    for et, pairs in edges.items():
        for s, t in pairs:
            g.add_edge(s, t)
            key_pairs.append((f"{s}::{t}::{et}", s, t))
    if g.number_of_edges() == 0:
        return {}
    ebc = nx.edge_betweenness_centrality(g)        # normalized, per node-pair
    raw = {}
    for key, s, t in key_pairs:
        raw[key] = ebc.get((s, t), ebc.get((t, s), 0.0))
    hi = max(raw.values()) or 1.0
    return {k: round(v / hi, 4) for k, v in raw.items()}   # min-max to [0,1]


def get_betweenness(edge_key: str) -> float:
    """Precomputed edge betweenness, normalized to [0,1]. 1 = strongest bridge."""
    global _BETWEENNESS_CACHE
    if _BETWEENNESS_CACHE is None:
        _BETWEENNESS_CACHE = _compute_all()
    return _BETWEENNESS_CACHE.get(edge_key, 0.0)


def reload_betweenness() -> None:
    global _BETWEENNESS_CACHE
    _BETWEENNESS_CACHE = None
