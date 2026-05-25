"""Persisted knowledge graph for Graph-RAG.

A `networkx.MultiDiGraph` whose nodes are normalised entity names and whose
edges carry relation type + provenance (which chunk_ids supplied the edge).
Persists to a node-link JSON file for offline inspection.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import networkx as nx

from config import GRAPH_PATH
from extractor import Entity, Extraction, Relation


def normalize(name: str) -> str:
    """Canonical entity key: trimmed and lowercased."""
    return " ".join(name.split()).lower()


class KnowledgeGraph:
    """Thin wrapper around networkx.MultiDiGraph with persistence + provenance."""

    def __init__(self, g: nx.MultiDiGraph | None = None) -> None:
        self.g: nx.MultiDiGraph = g if g is not None else nx.MultiDiGraph()

    def add_entity(self, ent: Entity, chunk_id: int) -> str:
        """Insert or merge an entity; record the supporting chunk_id."""
        key = normalize(ent.name)
        if not key:
            return key
        if self.g.has_node(key):
            node = self.g.nodes[key]
            node.setdefault("display", ent.name)
            if ent.type and ent.type != "unknown":
                node["type"] = ent.type
            if ent.description and ent.description not in node.get("descriptions", []):
                node.setdefault("descriptions", []).append(ent.description)
            if chunk_id not in node.setdefault("chunk_ids", []):
                node["chunk_ids"].append(chunk_id)
        else:
            self.g.add_node(
                key,
                display=ent.name,
                type=ent.type or "unknown",
                descriptions=[ent.description] if ent.description else [],
                chunk_ids=[chunk_id],
            )
        return key

    def add_relation(self, rel: Relation, chunk_id: int) -> bool:
        """Insert an edge; merge provenance if the same (src, dst, type) edge exists."""
        s, t = normalize(rel.source), normalize(rel.target)
        if not s or not t or s == t:
            return False
        # Auto-create endpoints if they're not yet known (extractor may emit
        # relations involving entities it didn't also list explicitly).
        for k, raw in ((s, rel.source), (t, rel.target)):
            if not self.g.has_node(k):
                self.g.add_node(k, display=raw, type="unknown", descriptions=[], chunk_ids=[chunk_id])

        for _, _, data in self.g.edges(s, data=True):
            pass  # placeholder, actual merge logic below

        # Find an existing parallel edge of the same type to merge into.
        for key, data in self.g[s].get(t, {}).items():
            if data.get("type") == rel.type:
                if rel.description and rel.description not in data.setdefault("descriptions", []):
                    data["descriptions"].append(rel.description)
                if chunk_id not in data.setdefault("chunk_ids", []):
                    data["chunk_ids"].append(chunk_id)
                return False

        self.g.add_edge(
            s,
            t,
            type=rel.type or "related_to",
            descriptions=[rel.description] if rel.description else [],
            chunk_ids=[chunk_id],
        )
        return True

    def ingest_extraction(self, extraction: Extraction, chunk_id: int) -> None:
        """Apply one chunk's extraction result to the graph."""
        for e in extraction.entities:
            self.add_entity(e, chunk_id)
        for r in extraction.relations:
            self.add_relation(r, chunk_id)

    def neighbors(self, node_key: str, hops: int = 1) -> set[str]:
        """Return all nodes within `hops` of `node_key` (treating the graph as undirected)."""
        if not self.g.has_node(node_key):
            return set()
        undirected = self.g.to_undirected(as_view=True)
        seen: set[str] = {node_key}
        frontier: set[str] = {node_key}
        for _ in range(max(0, hops)):
            nxt: set[str] = set()
            for n in frontier:
                nxt.update(undirected.neighbors(n))
            nxt -= seen
            if not nxt:
                break
            seen.update(nxt)
            frontier = nxt
        return seen - {node_key}

    def describe_node(self, key: str) -> str:
        """Compact text representation of a node for prompt injection."""
        if not self.g.has_node(key):
            return ""
        n = self.g.nodes[key]
        desc = "; ".join(n.get("descriptions", [])[:3])
        return f"[{n.get('type','?')}] {n.get('display', key)}: {desc}".strip(": ")

    def describe_edge(self, u: str, v: str) -> list[str]:
        """All edge descriptions between two nodes (both directions)."""
        out: list[str] = []
        for src, dst in ((u, v), (v, u)):
            if self.g.has_edge(src, dst):
                for _, data in self.g[src][dst].items():
                    s_disp = self.g.nodes[src].get("display", src)
                    t_disp = self.g.nodes[dst].get("display", dst)
                    out.append(f"{s_disp} --[{data.get('type','related_to')}]--> {t_disp}")
        return out

    def chunks_for_nodes(self, keys: Iterable[str]) -> set[int]:
        """Set of chunk_ids touched by any of the given nodes."""
        out: set[int] = set()
        for k in keys:
            if self.g.has_node(k):
                out.update(self.g.nodes[k].get("chunk_ids", []))
        return out

    def find_entities(self, query_text: str, limit: int = 8) -> list[str]:
        """Cheap lexical match: return node keys whose display/key occurs in the query."""
        q = query_text.lower()
        hits: list[tuple[int, str]] = []
        for key, data in self.g.nodes(data=True):
            disp = data.get("display", key).lower()
            if disp and disp in q:
                hits.append((len(disp), key))
            elif key in q:
                hits.append((len(key), key))
        hits.sort(reverse=True)
        return [k for _, k in hits[:limit]]

    def stats(self) -> dict:
        """Quick summary for /status."""
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges()}

    def save(self, path: Path = GRAPH_PATH) -> None:
        """Persist as networkx node-link JSON."""
        data = nx.node_link_data(self.g, edges="links")
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path = GRAPH_PATH) -> "KnowledgeGraph":
        """Load a previously saved graph, or return an empty one."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            g = nx.node_link_graph(data, directed=True, multigraph=True, edges="links")
            return cls(g)
        except Exception:
            return cls()
