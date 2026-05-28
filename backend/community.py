"""Community detection + per-community summarisation (Microsoft GraphRAG pattern).

Detects communities on the *undirected* projection of the knowledge graph
(networkx Louvain — Leiden is the gold standard but requires igraph/leidenalg;
Louvain is close enough and zero-dep beyond networkx) and asks DeepSeek to
summarise each community into a short report. Summaries are the backbone of
"global" queries: corpus-wide questions are answered by map-reducing across
community summaries instead of trying to stuff every chunk into one prompt.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import networkx as nx

from config import (
    COMMUNITIES_PATH,
    COMMUNITY_MAX_NODES,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_INGEST_MODEL,
    DEEPSEEK_TIMEOUT,
)
from graph import KnowledgeGraph


COMMUNITY_SYSTEM = (
    "You are summarising one community of a knowledge graph for a Graph-RAG system. "
    "Given a list of entities (with types and descriptions) and the relations between "
    "them, write a concise report in 3 sections:\n"
    "TITLE: a 4-8 word name for the community.\n"
    "SUMMARY: 2-4 sentences describing what binds these entities together.\n"
    "KEY FINDINGS: 3-6 bullets of the most important facts or claims supported "
    "by the relations.\n"
    "Be specific. Do not invent facts not present in the input."
)


@dataclass
class Community:
    """One detected community with its summary."""
    id: int
    nodes: list[str] = field(default_factory=list)
    title: str = ""
    summary: str = ""
    findings: list[str] = field(default_factory=list)
    size: int = 0

    def to_dict(self) -> dict:
        """Serialise to a JSON-friendly dict."""
        return {
            "id": self.id,
            "nodes": self.nodes,
            "title": self.title,
            "summary": self.summary,
            "findings": self.findings,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Community":
        """Inverse of to_dict."""
        return cls(
            id=int(d["id"]),
            nodes=list(d.get("nodes", [])),
            title=d.get("title", ""),
            summary=d.get("summary", ""),
            findings=list(d.get("findings", [])),
            size=int(d.get("size", len(d.get("nodes", [])))),
        )


def detect_communities(kg: KnowledgeGraph) -> list[list[str]]:
    """Run Louvain community detection on the undirected projection."""
    if kg.g.number_of_nodes() == 0:
        return []
    ug = nx.Graph()
    ug.add_nodes_from(kg.g.nodes(data=True))
    for u, v, data in kg.g.edges(data=True):
        if ug.has_edge(u, v):
            ug[u][v]["weight"] += 1.0
        else:
            ug.add_edge(u, v, weight=1.0)
    # networkx 3.x: louvain_communities returns a list of sets.
    parts = list(nx.community.louvain_communities(ug, seed=42, weight="weight"))
    return [sorted(p) for p in sorted(parts, key=len, reverse=True)]


def _community_prompt_body(kg: KnowledgeGraph, nodes: list[str]) -> str:
    """Render the entity+relation block sent to DeepSeek."""
    nodes = nodes[:COMMUNITY_MAX_NODES]
    lines = ["ENTITIES:"]
    for k in nodes:
        d = kg.describe_node(k)
        if d:
            lines.append(f"- {d}")
    lines.append("\nRELATIONS:")
    nodeset = set(nodes)
    edge_lines: list[str] = []
    for u, v, data in kg.g.edges(data=True):
        if u in nodeset and v in nodeset:
            u_disp = kg.g.nodes[u].get("display", u)
            v_disp = kg.g.nodes[v].get("display", v)
            edge_lines.append(f"- {u_disp} --[{data.get('type','related_to')}]--> {v_disp}")
    if not edge_lines:
        edge_lines.append("- (no within-community edges)")
    lines.extend(edge_lines)
    return "\n".join(lines)


def _parse_summary(text: str) -> tuple[str, str, list[str]]:
    """Pull TITLE / SUMMARY / KEY FINDINGS sections out of free-form output."""
    title, summary, findings = "", "", []
    current = None
    buf: list[str] = []

    def flush():
        nonlocal title, summary
        if current == "title":
            title = " ".join(buf).strip()
        elif current == "summary":
            summary = " ".join(buf).strip()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.upper().startswith("TITLE:"):
            flush(); buf = [line.split(":", 1)[1].strip()]; current = "title"
        elif line.upper().startswith("SUMMARY:"):
            flush(); buf = [line.split(":", 1)[1].strip()]; current = "summary"
        elif line.upper().startswith("KEY FINDINGS"):
            flush(); buf = []; current = "findings"
        elif current == "findings" and (line.startswith("-") or line.startswith("•") or line.startswith("*")):
            findings.append(line.lstrip("-•* ").strip())
        elif current in ("title", "summary"):
            buf.append(line)
    flush()
    return title, summary, [f for f in findings if f]


def summarise_community(kg: KnowledgeGraph, nodes: list[str], *, client: httpx.Client) -> tuple[str, str, list[str]]:
    """Ask DeepSeek to title + summarise + list findings for one community."""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not set.")
    body = _community_prompt_body(kg, nodes)
    payload = {
        "model": DEEPSEEK_INGEST_MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": COMMUNITY_SYSTEM},
            {"role": "user", "content": body},
        ],
    }
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    r = client.post(f"{DEEPSEEK_BASE_URL.rstrip('/')}/v1/chat/completions", json=payload, headers=headers)
    r.raise_for_status()
    text = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    return _parse_summary(text)


def build_communities(kg: KnowledgeGraph, *, progress: bool = True) -> list[Community]:
    """Detect communities and (optionally) generate LLM summaries for each.

    Set env SKIP_COMMUNITY_SUMMARIES=1 to keep the community detection but
    skip the per-community LLM call — useful for fast ingest where global
    graph mode is acceptable with placeholder titles.
    """
    parts = detect_communities(kg)
    communities: list[Community] = []
    skip = os.environ.get("SKIP_COMMUNITY_SUMMARIES", "") == "1"
    if skip:
        # Detection-only path — never opens an httpx client, never hits DeepSeek.
        for i, nodes in enumerate(parts):
            if not nodes:
                continue
            communities.append(
                Community(id=i, nodes=nodes, title="", summary="", findings=[], size=len(nodes))
            )
            if progress:
                print(f"  [community:skip-summary] {i}: {len(nodes)} nodes")
        return communities
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Parallel community summarisation — same 4-worker pool as chunk extraction.
    def _summarise_one(i: int, nodes: list[str]) -> tuple[int, str, str, list[str], list[str], str | None]:
        if len(nodes) < 2:
            return (i, "", "", [], nodes, None)
        try:
            with httpx.Client(timeout=DEEPSEEK_TIMEOUT) as cl:
                title, summary, findings = summarise_community(kg, nodes, client=cl)
            return (i, title, summary, findings, nodes, None)
        except Exception as e:
            return (i, "", "", [], nodes, str(e))

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_summarise_one, i, nodes): i
            for i, nodes in enumerate(parts) if nodes
        }
        results_by_id: dict[int, tuple] = {}
        done_count = 0
        total = len(futures)
        for future in as_completed(futures):
            done_count += 1
            i, title, summary, findings, nodes, err = future.result()
            results_by_id[i] = (title, summary, findings, nodes, err)
            if progress:
                tname = title or f"community-{i}"
                status = f" — {tname}" if title else (" (error)" if err else "")
                if done_count % max(1, total // 10) == 0 or done_count == total:
                    print(f"  [community] {done_count}/{total}{status}")

    for i in sorted(results_by_id):
        title, summary, findings, nodes, err = results_by_id[i]
        if err and progress:
            print(f"  [community:warn] {i}: {err}")
        communities.append(
            Community(id=i, nodes=nodes, title=title, summary=summary, findings=findings, size=len(nodes))
        )

    return communities


def save_communities(communities: list[Community], path: Path = COMMUNITIES_PATH) -> None:
    """Persist communities to disk."""
    path.write_text(json.dumps([c.to_dict() for c in communities], indent=2, ensure_ascii=False))


def load_communities(path: Path = COMMUNITIES_PATH) -> list[Community]:
    """Load communities; return [] if missing."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return [Community.from_dict(d) for d in data]
    except Exception:
        return []


def community_index_by_node(communities: list[Community]) -> dict[str, int]:
    """Map node_key → community id for fast lookup."""
    idx: dict[str, int] = {}
    for c in communities:
        for n in c.nodes:
            idx[n] = c.id
    return idx
