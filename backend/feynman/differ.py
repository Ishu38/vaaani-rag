"""Core Feynman-diff logic.

Pipeline:
  1. Load the corpus graph (data/graph.json — NetworkX node-link JSON).
  2. Pick a topic node and BFS k hops to get the "target subgraph".
  3. Extract entities + relations from the student's free-form explanation
     (reuses extractor.extract_chunk → DeepSeek JSON mode).
  4. Match student-named entities back to graph nodes via case-insensitive
     name comparison + substring fallback.
  5. Set-diff: nodes_covered, nodes_missed, edges_covered, edges_missed.

No embedding similarity in v1 — exact + substring matching gets us most
of the way and keeps the round-trip under one DeepSeek call.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from config import GRAPH_PATH
from extractor import Extraction, Entity, Relation, extract_chunk


# =========================================================================
#  Graph loading
# =========================================================================

@dataclass
class _GraphView:
    nodes: dict[str, dict]                # id -> node dict
    adj: dict[str, set[str]]              # id -> set of neighbour ids
    links: list[dict]                     # raw edge list (with source/target ids)


def _load_graph_view() -> _GraphView:
    if not Path(GRAPH_PATH).exists():
        return _GraphView(nodes={}, adj=defaultdict(set), links=[])
    raw = json.loads(Path(GRAPH_PATH).read_text())
    nodes_by_id: dict[str, dict] = {}
    for n in raw.get("nodes", []):
        nid = str(n.get("id", "")).strip()
        if not nid:
            continue
        nodes_by_id[nid] = n
    adj: dict[str, set[str]] = defaultdict(set)
    links: list[dict] = []
    for e in raw.get("links", []) or raw.get("edges", []):
        s = str(e.get("source", "")).strip()
        t = str(e.get("target", "")).strip()
        if not s or not t or s not in nodes_by_id or t not in nodes_by_id:
            continue
        adj[s].add(t)
        adj[t].add(s)  # undirected adjacency for BFS / diff purposes
        links.append(e)
    return _GraphView(nodes=nodes_by_id, adj=adj, links=links)


# =========================================================================
#  Topic selection
# =========================================================================

def list_topics(min_degree: int = 2, limit: int = 60) -> list[dict]:
    """Topics worth picking: nodes with at least min_degree neighbours,
    ranked by degree desc so the densest concepts surface first.
    """
    g = _load_graph_view()
    scored: list[tuple[int, str]] = []
    for nid in g.nodes:
        deg = len(g.adj.get(nid, ()))
        if deg >= min_degree:
            scored.append((deg, nid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out: list[dict] = []
    for deg, nid in scored[:limit]:
        n = g.nodes[nid]
        descriptions = n.get("descriptions") or []
        desc = descriptions[0] if descriptions else ""
        out.append({
            "id": nid,
            "display": n.get("display", nid),
            "type": n.get("type", ""),
            "degree": deg,
            "description": desc[:160],
        })
    return out


# =========================================================================
#  Subgraph selection
# =========================================================================

def _subgraph(g: _GraphView, target_id: str, k: int) -> tuple[set[str], list[dict]]:
    """BFS k hops from target_id. Returns (node_ids, edges_within_subgraph)."""
    if target_id not in g.nodes:
        return set(), []
    visited: dict[str, int] = {target_id: 0}
    frontier: deque[str] = deque([target_id])
    while frontier:
        cur = frontier.popleft()
        if visited[cur] >= k:
            continue
        for nb in g.adj.get(cur, ()):
            if nb not in visited:
                visited[nb] = visited[cur] + 1
                frontier.append(nb)
    node_ids = set(visited.keys())
    sub_edges = [
        e for e in g.links
        if e.get("source") in node_ids and e.get("target") in node_ids
    ]
    return node_ids, sub_edges


# =========================================================================
#  Student → graph entity matching
# =========================================================================

_NORMALIZE_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = _NORMALIZE_RE.sub(" ", s)
    return " ".join(s.split())


def _match_entity_to_node(
    student_name: str,
    candidate_ids: set[str],
    g: _GraphView,
) -> str | None:
    """Resolve a student-mentioned entity name to a node id within the
    candidate set. Exact match wins; otherwise allow substring overlap on
    normalised forms with a minimum 3-char overlap to skip junk matches.
    """
    target = _normalize(student_name)
    if not target:
        return None
    # 1. Exact match on display name or id.
    for nid in candidate_ids:
        n = g.nodes[nid]
        if _normalize(n.get("display", "")) == target or _normalize(nid) == target:
            return nid
    # 2. Substring containment (either direction). Require min 3 char target
    #    to avoid matching things like "it" or "is" to graph nodes.
    if len(target) < 3:
        return None
    for nid in candidate_ids:
        n = g.nodes[nid]
        cand = _normalize(n.get("display", ""))
        if not cand:
            continue
        if target in cand or cand in target:
            return nid
    return None


# =========================================================================
#  Diff result
# =========================================================================

@dataclass
class _NodeView:
    id: str
    display: str
    type: str
    description: str


@dataclass
class _EdgeView:
    source: str
    target: str
    source_display: str
    target_display: str
    type: str
    description: str


@dataclass
class FeynmanResult:
    topic_id: str
    topic_display: str
    k_hops: int
    nodes_in_subgraph: int
    edges_in_subgraph: int
    coverage_pct: float
    nodes_covered: list[_NodeView] = field(default_factory=list)
    nodes_missed: list[_NodeView] = field(default_factory=list)
    edges_covered: list[_EdgeView] = field(default_factory=list)
    edges_missed: list[_EdgeView] = field(default_factory=list)
    student_extras: list[str] = field(default_factory=list)  # named entities not in subgraph
    summary: str = ""                                        # plain-language overview

    def to_json(self) -> dict:
        d = asdict(self)
        return d


def _node_view(g: _GraphView, nid: str) -> _NodeView:
    n = g.nodes.get(nid, {})
    descriptions = n.get("descriptions") or []
    return _NodeView(
        id=nid,
        display=n.get("display", nid),
        type=n.get("type", ""),
        description=(descriptions[0] if descriptions else "")[:200],
    )


def _edge_view(g: _GraphView, e: dict) -> _EdgeView:
    s = e.get("source", "")
    t = e.get("target", "")
    descriptions = e.get("descriptions") or []
    return _EdgeView(
        source=s,
        target=t,
        source_display=g.nodes.get(s, {}).get("display", s),
        target_display=g.nodes.get(t, {}).get("display", t),
        type=e.get("type", "related_to"),
        description=(descriptions[0] if descriptions else "")[:200],
    )


# =========================================================================
#  Main entry point
# =========================================================================

def diff_explanation(student_text: str, target_node_id: str, *, k: int = 2) -> FeynmanResult:
    """Run the full Feynman pipeline. Raises ValueError for bad input,
    KeyError if the topic node doesn't exist in the graph."""
    if not student_text.strip():
        raise ValueError("explanation is empty")
    g = _load_graph_view()
    if target_node_id not in g.nodes:
        raise KeyError(f"unknown topic '{target_node_id}'")

    sub_node_ids, sub_edges = _subgraph(g, target_node_id, k)

    # Run DeepSeek extractor once on the student's text.
    ext: Extraction = extract_chunk(student_text)

    # Resolve each student entity to a subgraph node (or None → extras).
    matched_nodes: set[str] = set()
    extras: list[str] = []
    student_name_to_node: dict[str, str | None] = {}
    for ent in ext.entities:
        node_id = _match_entity_to_node(ent.name, sub_node_ids, g)
        student_name_to_node[ent.name] = node_id
        if node_id:
            matched_nodes.add(node_id)
        elif len(ent.name.strip()) >= 3:
            extras.append(ent.name)

    # Edge matching. An edge (a,b) in the subgraph is "covered" if the
    # student mentioned BOTH a relation between something resolving to a
    # and something resolving to b (in either direction).
    student_edge_pairs: set[frozenset[str]] = set()
    for rel in ext.relations:
        s_id = student_name_to_node.get(rel.source) or _match_entity_to_node(
            rel.source, sub_node_ids, g
        )
        t_id = student_name_to_node.get(rel.target) or _match_entity_to_node(
            rel.target, sub_node_ids, g
        )
        if s_id and t_id and s_id != t_id:
            student_edge_pairs.add(frozenset({s_id, t_id}))
            # The student named both endpoints, so they're covered even if
            # the entity itself wasn't mentioned standalone.
            matched_nodes.add(s_id)
            matched_nodes.add(t_id)

    nodes_covered: list[_NodeView] = []
    nodes_missed: list[_NodeView] = []
    for nid in sub_node_ids:
        view = _node_view(g, nid)
        if nid in matched_nodes:
            nodes_covered.append(view)
        else:
            nodes_missed.append(view)

    edges_covered: list[_EdgeView] = []
    edges_missed: list[_EdgeView] = []
    for e in sub_edges:
        pair = frozenset({e.get("source", ""), e.get("target", "")})
        if pair in student_edge_pairs:
            edges_covered.append(_edge_view(g, e))
        else:
            edges_missed.append(_edge_view(g, e))

    total_nodes = max(len(sub_node_ids), 1)
    pct = round(100.0 * len(nodes_covered) / total_nodes, 1)

    # Sort outputs so the UI is stable.
    nodes_covered.sort(key=lambda n: n.display.lower())
    nodes_missed.sort(key=lambda n: n.display.lower())
    edges_covered.sort(key=lambda e: (e.source_display.lower(), e.target_display.lower()))
    edges_missed.sort(key=lambda e: (e.source_display.lower(), e.target_display.lower()))

    return FeynmanResult(
        topic_id=target_node_id,
        topic_display=g.nodes[target_node_id].get("display", target_node_id),
        k_hops=k,
        nodes_in_subgraph=len(sub_node_ids),
        edges_in_subgraph=len(sub_edges),
        coverage_pct=pct,
        nodes_covered=nodes_covered,
        nodes_missed=nodes_missed,
        edges_covered=edges_covered,
        edges_missed=edges_missed,
        student_extras=sorted(set(extras))[:20],
        summary=summarize_diff(
            topic=g.nodes[target_node_id].get("display", target_node_id),
            covered_pct=pct,
            n_missed_nodes=len(nodes_missed),
            n_missed_edges=len(edges_missed),
            sample_missed_node=nodes_missed[0].display if nodes_missed else "",
            sample_missed_edge=(
                f"{edges_missed[0].source_display} → {edges_missed[0].target_display}"
                if edges_missed else ""
            ),
        ),
    )


def summarize_diff(
    *,
    topic: str,
    covered_pct: float,
    n_missed_nodes: int,
    n_missed_edges: int,
    sample_missed_node: str,
    sample_missed_edge: str,
) -> str:
    """Plain-language one-liner. Deterministic (no LLM call) so this is
    free to compute; the structured lists below it carry the detail."""
    if covered_pct >= 90 and n_missed_edges <= 1:
        return f"You covered {covered_pct}% of the {topic} subgraph — strong explanation."
    bits: list[str] = [f"You covered {covered_pct}% of the {topic} subgraph"]
    if n_missed_nodes and sample_missed_node:
        bits.append(f"missed {n_missed_nodes} concept{'s' if n_missed_nodes != 1 else ''} including {sample_missed_node}")
    if n_missed_edges and sample_missed_edge:
        bits.append(f"and {n_missed_edges} connection{'s' if n_missed_edges != 1 else ''} including {sample_missed_edge}")
    return ", ".join(bits) + "."
