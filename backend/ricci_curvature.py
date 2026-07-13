"""Ricci Curvature — zero-training-data difficulty prior for CASCADE.

Computes Forman-Ricci curvature κ(e) for every edge in the linguistic graph.
High κ = dense neighborhood (easy edge); low κ = sparse/bottleneck (hard edge).

Normalized to [0,1] where 0 = bottleneck (most difficult) and 1 = dense (easiest).
This gives the CASCADE planner an intrinsic difficulty prior BEFORE seeing
any learner data — graph topology alone predicts which edges are hard.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from config import GRAPH_PATH
except ImportError:
    GRAPH_PATH = Path(__file__).resolve().parent.parent / "data" / "graph.json"

_CURVATURE_CACHE: dict[str, float] | None = None


def _compute_all_curvatures() -> dict[str, float]:
    """Precompute Forman-Ricci curvature for every edge. Called once."""
    from graph import KnowledgeGraph
    kg = KnowledgeGraph()
    nxg = kg.g  # networkx MultiDiGraph

    curvatures: dict[str, float] = {}
    for u, v, data in nxg.edges(data=True):
        etype = data.get("type", "related_to")
        key = f"{u}::{v}::{etype}"

        # Forman-Ricci: κ(u,v) = 4 - deg(u) - deg(v)
        # For directed multigraph, use out-degree for u and in-degree for v
        deg_u = nxg.degree(u)
        deg_v = nxg.degree(v)
        raw = 4.0 - deg_u - deg_v

        # Sigmoid normalization to [0,1]
        # raw ∈ [-N, 4] typically; center at 0
        normalized = 1.0 / (1.0 + __import__("math").exp(-raw / 3.0))
        curvatures[key] = round(normalized, 4)

    return curvatures


def get_curvature(edge_key: str) -> float:
    """Get precomputed Forman-Ricci curvature for an edge. [0,1]."""
    global _CURVATURE_CACHE
    if _CURVATURE_CACHE is None:
        _CURVATURE_CACHE = _compute_all_curvatures()
    return _CURVATURE_CACHE.get(edge_key, 0.5)


def reload_curvatures() -> None:
    """Force recomputation (after graph seed changes)."""
    global _CURVATURE_CACHE
    _CURVATURE_CACHE = None
