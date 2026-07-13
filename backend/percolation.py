"""Percolation Mastery — CASCADE emergent node mastery.

A node is mastered not when its direct BKT exceeds a threshold, but
when the learned-edge subgraph forms a connected component large enough
that a random walk cannot escape. This is the CASCADE insight: language
knowledge is measured by the density of learned relationships, not by
isolated word counts.

Formal definition:
  Let G = (V, E) be the full linguistic graph.
  Let L = {e ∈ E : mastery(e) ≥ θ} be the learned edges at threshold θ.
  Let G_L = (V, L) be the subgraph of learned edges.
  Node v is PERCOLATED if v is in a connected component of G_L
  with at least k_min nodes.

This replaces the node-level `B(v) ≥ MASTERED_AT` criterion with
an edge-topological emergence criterion.
"""

from __future__ import annotations

import math
import random


def percolated_nodes(
    edge_beliefs: dict,
    graph_edges: dict,
    theta: float = 0.90,
    k_min: int = 3,
) -> set[str]:
    """Return the set of nodes mastered via percolation.

    A node is percolated (mastered) if it belongs to a connected
    component of the learned-edge subgraph with at least k_min nodes.

    Args:
        edge_beliefs: {edge_key: EdgeBelief}
        graph_edges: {etype: [(source, target), ...]}
        theta: mastery threshold for an edge to be 'learned'
        k_min: minimum component size for percolation

    Returns:
        set of percolated node IDs
    """
    # Build adjacency from learned edges
    adj: dict[str, set[str]] = {}
    for etype, pairs in graph_edges.items():
        for s, t in pairs:
            key = f"{s}::{t}::{etype}"
            bel = edge_beliefs.get(key)
            if bel is not None and bel.mastery >= theta:
                adj.setdefault(s, set()).add(t)
                adj.setdefault(t, set()).add(s)

    # Find connected components
    visited: set[str] = set()
    percolated: set[str] = set()

    for node in adj:
        if node in visited:
            continue
        # BFS to find component
        component: set[str] = set()
        queue = [node]
        while queue:
            v = queue.pop()
            if v in component:
                continue
            component.add(v)
            queue.extend(adj.get(v, set()) - component)
        visited.update(component)
        if len(component) >= k_min:
            percolated.update(component)

    return percolated


def percolation_probability(
    node: str,
    edge_beliefs: dict,
    graph_edges: dict,
    walk_length: int = 3,
    num_walks: int = 100,
) -> float:
    """Estimate the percolation probability for one node.

    Simulates random walks starting at `node`. At each step, follows
    a random incident edge with probability proportional to its mastery.
    Returns fraction of walks that stay within nodes that have at least
    one incident edge (i.e., the walk doesn't hit a dead end).

    P_perc(v) ≈ fraction of walks surviving k steps without starvation.
    """
    if num_walks <= 0:
        return 0.0

    neighbors: dict[str, list[tuple[str, float]]] = {}
    for etype, pairs in graph_edges.items():
        for s, t in pairs:
            key = f"{s}::{t}::{etype}"
            bel = edge_beliefs.get(key)
            w = bel.mastery if bel else 0.1
            neighbors.setdefault(s, []).append((t, w))
            neighbors.setdefault(t, []).append((s, w))

    if node not in neighbors:
        return 0.0

    survivors = 0
    rng = random.Random(node)  # deterministic seed per node

    for _ in range(num_walks):
        current = node
        alive = True
        for _ in range(walk_length):
            nbrs = neighbors.get(current, [])
            if not nbrs:
                alive = False
                break
            # Weighted random choice by edge mastery
            total_w = sum(w for _, w in nbrs)
            if total_w <= 0:
                alive = False
                break
            r = rng.random() * total_w
            cumulative = 0.0
            chosen = nbrs[0][0]
            for nxt, w in nbrs:
                cumulative += w
                if r <= cumulative:
                    chosen = nxt
                    break
            current = chosen
        if alive:
            survivors += 1

    return survivors / num_walks


def is_mastered_by_percolation(
    node: str,
    edge_beliefs: dict,
    graph_edges: dict,
    theta_perc: float = 0.85,
    theta_edge: float = 0.90,
) -> bool:
    """Mastered iff the percolation probability exceeds theta_perc.

    This is the continuous version — a node can be 'nearly mastered'
    (percolation_prob ≈ 0.80) before the binary threshold kicks in.
    """
    # Fast check: already in a large learned component?
    learned = percolated_nodes(edge_beliefs, graph_edges,
                               theta=theta_edge, k_min=3)
    if node in learned:
        return True

    # Slow check: simulate random walks
    p = percolation_probability(node, edge_beliefs, graph_edges,
                                 walk_length=3, num_walks=50)
    return p >= theta_perc
