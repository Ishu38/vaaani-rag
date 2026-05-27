"""Retrieval layer.

Wraps the TurboQuantIndex + metadata sidecar + sentence-transformer embedder.
Exposes a singleton-style `Retriever` so the FastAPI process loads the model
exactly once.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer
from turbovec import TurboQuantIndex

from community import Community, community_index_by_node, load_communities
from config import (
    BIT_WIDTH,
    COMMUNITIES_PATH,
    EMBED_DIM,
    EMBED_MODEL_NAME,
    GLOBAL_TOP_COMMUNITIES,
    GRAPH_PATH,
    INDEX_PATH,
    LOCAL_HOPS,
    METADATA_PATH,
    TOP_K,
)
from graph import KnowledgeGraph


class Retriever:
    """Lazy-loaded retriever: embeds queries and searches the TurboVec index."""

    def __init__(
        self,
        index_path: Path = INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
        model_name: str = EMBED_MODEL_NAME,
    ) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._index: TurboQuantIndex | None = None
        self._metadata: dict | None = None
        self._kg: KnowledgeGraph | None = None
        self._communities: list[Community] | None = None
        self._community_idx: dict[str, int] | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Cached embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def index(self) -> TurboQuantIndex | None:
        """Cached TurboVec index, or None if no index has been built yet."""
        if self._index is None and self.index_path.exists():
            self._index = TurboQuantIndex.load(str(self.index_path))
        return self._index

    @property
    def metadata(self) -> dict:
        """Cached metadata sidecar."""
        if self._metadata is None:
            if self.metadata_path.exists():
                self._metadata = json.loads(self.metadata_path.read_text())
            else:
                self._metadata = {"files": {}, "chunks": []}
        return self._metadata

    @property
    def kg(self) -> KnowledgeGraph:
        """Cached knowledge graph (empty if no graph has been built)."""
        if self._kg is None:
            self._kg = KnowledgeGraph.load(GRAPH_PATH)
        return self._kg

    @property
    def communities(self) -> list[Community]:
        """Cached community list (empty if none built)."""
        if self._communities is None:
            self._communities = load_communities(COMMUNITIES_PATH)
        return self._communities

    @property
    def community_idx(self) -> dict[str, int]:
        """Cached node_key → community_id index."""
        if self._community_idx is None:
            self._community_idx = community_index_by_node(self.communities)
        return self._community_idx

    def reload(self) -> None:
        """Invalidate caches after a re-ingest."""
        self._index = None
        self._metadata = None
        self._kg = None
        self._communities = None
        self._community_idx = None

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        """Embed a list of strings to a (n, dim) float32 array."""
        vecs = self.model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)

    def search(self, query: str, k: int = TOP_K, *, source_filter: set[str] | None = None) -> list[dict]:
        """Return up to `k` chunks as [{text, source, score}] for `query`.

        Source-filename overlap boost: when the query contains tokens that
        appear in a chunk's source filename, that chunk's score is
        multiplied by SOURCE_HIT_BOOST.

        Hard source filter: if ``source_filter`` is a non-empty set, only
        chunks whose ``source`` field is in that set are returned. This
        backs the per-document "scope chip" in the UI.
        """
        if not query.strip() or self.index is None:
            return []
        chunks = self.metadata.get("chunks", [])
        if not chunks:
            return []
        qv = np.ascontiguousarray(self.embed([query]), dtype=np.float32)
        # Pull a wider candidate set so the source-boost re-ranking has room.
        wide_k = min(max(k * 4, k), len(chunks))
        scores, indices = self.index.search(qv, k=wide_k)
        if hasattr(scores, "ndim") and scores.ndim == 2:
            scores = scores[0]
            indices = indices[0]

        # Tokenise the query into significant words (≥4 chars, alphabetic).
        # Short words like "the", "of", "in" would match every PDF filename
        # and defeat the point of the boost.
        import re as _re
        query_tokens = {
            t.lower() for t in _re.findall(r"[A-Za-z]{4,}", query)
        }
        SOURCE_HIT_BOOST = 1.35  # +35% to chunks whose source filename overlaps query

        out: list[dict] = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(chunks):
                continue
            ch = chunks[idx]
            # Hard scope filter — drop chunks not in the allowed source set.
            if source_filter and ch["source"] not in source_filter:
                continue
            adj = float(score)
            if query_tokens:
                src_tokens = {
                    t.lower() for t in _re.findall(r"[A-Za-z]{4,}", ch["source"])
                }
                if query_tokens & src_tokens:
                    adj *= SOURCE_HIT_BOOST
            out.append({
                "text": ch["text"],
                "source": ch["source"],
                "score": adj,
            })

        # Re-rank by adjusted score, trim to k.
        out.sort(key=lambda r: r["score"], reverse=True)
        return out[:k]

    def chunk_by_id(self, chunk_id: int) -> dict | None:
        """Return the metadata for a given chunk_id, or None if out of range."""
        chunks = self.metadata.get("chunks", [])
        if 0 <= chunk_id < len(chunks):
            ch = chunks[chunk_id]
            return {"text": ch["text"], "source": ch["source"], "chunk_id": chunk_id}
        return None

    def local_graph_search(self, query: str, k: int = TOP_K, *, source_filter: set[str] | None = None) -> dict:
        """Local Graph-RAG: vector hits + 1-hop entity neighbourhood + community context.

        Returns a dict with:
          - chunks: union of vector hits and chunks linked through the entity graph
          - entities: matched + neighbour entity display names
          - communities: at most one community summary for the dominant matched community
          - edges: human-readable edge descriptions among matched entities

        ``source_filter`` (optional): set of source filenames to restrict
        retrieval to. Passes through to ``search()`` and additionally
        drops graph-expanded chunks whose source isn't in the set.
        """
        vector_hits = self.search(query, k=k, source_filter=source_filter)
        chunks_by_id: dict[int, dict] = {}
        # seed with vector hits (assign synthetic ids from metadata position)
        kg = self.kg
        chunks_meta = self.metadata.get("chunks", [])
        for h in vector_hits:
            # locate chunk_id by scanning metadata for an exact match (vector order
            # mirrors insertion order, so this is rarely needed — keep it robust).
            for idx, ch in enumerate(chunks_meta):
                if ch["text"] == h["text"] and ch["source"] == h["source"]:
                    chunks_by_id[idx] = {**h, "chunk_id": idx}
                    break

        # Find entities mentioned in the query, expand neighborhood, gather their chunks.
        matched = kg.find_entities(query)
        neighbour_keys: set[str] = set(matched)
        for k_ in matched:
            neighbour_keys |= kg.neighbors(k_, hops=LOCAL_HOPS)
        for cid in kg.chunks_for_nodes(neighbour_keys):
            ch = self.chunk_by_id(cid)
            if ch is None or cid in chunks_by_id:
                continue
            # Scope-chip respected on graph-expansion too — otherwise the
            # filter leaks via the entity neighborhood.
            if source_filter and ch["source"] not in source_filter:
                continue
            chunks_by_id[cid] = {**ch, "score": 0.0}

        # Edge descriptions among matched entities (small N).
        edges: list[str] = []
        m = list(matched)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                edges.extend(kg.describe_edge(m[i], m[j]))

        # Dominant community — restricted to the source documents that
        # actually answered the query. Previously this counted nodes across
        # the whole corpus's communities, which surfaced a community from a
        # different book when that book happened to be larger. Now we
        # derive per-node sources from chunk_ids → chunks_meta[i]["source"]
        # and only count nodes whose sources overlap the answering set.
        answering_sources = {ch.get("source") for ch in chunks_by_id.values() if ch.get("source")}

        def _node_sources(nk: str) -> set[str]:
            if not kg.g.has_node(nk):
                return set()
            cids = kg.g.nodes[nk].get("chunk_ids", []) or []
            return {
                chunks_meta[c]["source"]
                for c in cids
                if 0 <= c < len(chunks_meta) and "source" in chunks_meta[c]
            }

        comm_counts: dict[int, int] = {}
        for nk in neighbour_keys:
            cid = self.community_idx.get(nk)
            if cid is None:
                continue
            if answering_sources:
                ns = _node_sources(nk)
                if ns and not (ns & answering_sources):
                    # Node lives entirely outside the answering sources →
                    # don't let it sway the dominant-community vote.
                    continue
            comm_counts[cid] = comm_counts.get(cid, 0) + 1

        communities: list[Community] = []
        if comm_counts:
            top_cid = max(comm_counts, key=comm_counts.get)
            for c in self.communities:
                if c.id == top_cid:
                    communities = [c]
                    break

        entities = [kg.g.nodes[k_].get("display", k_) for k_ in neighbour_keys if kg.g.has_node(k_)]
        return {
            "chunks": list(chunks_by_id.values())[: max(k, len(vector_hits)) + LOCAL_HOPS * 3],
            "entities": entities[:12],
            "communities": communities,
            "edges": edges[:8],
        }

    def global_graph_search(self, query: str) -> dict:
        """Global Graph-RAG: rank community summaries by semantic relevance to the query."""
        comms = self.communities
        if not comms:
            return {"chunks": [], "entities": [], "communities": [], "edges": []}
        # Embed query + community summaries; pick top-k communities by cosine.
        summaries: list[str] = []
        valid: list[Community] = []
        for c in comms:
            blob = f"{c.title}\n{c.summary}\n" + "\n".join(c.findings)
            if blob.strip():
                summaries.append(blob)
                valid.append(c)
        if not summaries:
            return {"chunks": [], "entities": [], "communities": comms[:GLOBAL_TOP_COMMUNITIES], "edges": []}
        qv = self.embed([query])[0]
        sv = self.embed(summaries)
        qv_n = qv / (np.linalg.norm(qv) + 1e-12)
        sv_n = sv / (np.linalg.norm(sv, axis=1, keepdims=True) + 1e-12)
        scores = sv_n @ qv_n
        order = np.argsort(-scores)[:GLOBAL_TOP_COMMUNITIES]
        picked = [valid[i] for i in order]
        # Pull a few representative chunks from the top community for grounding.
        chunks: list[dict] = []
        if picked:
            top_nodes = picked[0].nodes[:10]
            for cid in self.kg.chunks_for_nodes(top_nodes):
                ch = self.chunk_by_id(cid)
                if ch is not None:
                    chunks.append({**ch, "score": float(scores[order[0]])})
                if len(chunks) >= TOP_K:
                    break
        return {"chunks": chunks, "entities": [], "communities": picked, "edges": []}

    def status(self) -> dict:
        """Return a status summary for the /status endpoint."""
        total_chunks = len(self.metadata.get("chunks", []))
        index_size_mb = (
            round(self.index_path.stat().st_size / (1024 * 1024), 3)
            if self.index_path.exists()
            else 0.0
        )
        docs = [v["name"] for v in self.metadata.get("files", {}).values()]
        return {
            "total_chunks": total_chunks,
            "index_size_mb": index_size_mb,
            "documents_indexed": docs,
            "embedding_dim": EMBED_DIM,
            "bit_width": BIT_WIDTH,
            "graph_nodes": self.kg.g.number_of_nodes(),
            "graph_edges": self.kg.g.number_of_edges(),
            "communities_count": len(self.communities),
        }
