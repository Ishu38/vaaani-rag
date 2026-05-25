"""Rolling JSON memory layer.

Stores long-term facts and a bounded queue of recent queries. Surfaces the
top-K most relevant facts for any given query using cosine similarity over
sentence-transformer embeddings (computed on demand; cheap at small scale).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from config import MAX_RECENT_QUERIES, MEMORY_PATH, MEMORY_TOP_K


def _empty() -> dict:
    """Return the canonical empty memory structure."""
    return {"facts": [], "recent_queries": []}


def load_memory(path: Path = MEMORY_PATH) -> dict:
    """Load memory.json, creating a fresh skeleton if it doesn't exist."""
    if not path.exists():
        return _empty()
    try:
        data = json.loads(path.read_text())
        data.setdefault("facts", [])
        data.setdefault("recent_queries", [])
        return data
    except Exception:
        return _empty()


def save_memory(mem: dict, path: Path = MEMORY_PATH) -> None:
    """Persist memory.json atomically (write-then-rename)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(mem, indent=2, ensure_ascii=False))
    tmp.replace(path)


def add_fact(fact: str, path: Path = MEMORY_PATH) -> dict:
    """Append a fact (deduped) and return the updated memory."""
    mem = load_memory(path)
    if fact and fact not in mem["facts"]:
        mem["facts"].append(fact)
        save_memory(mem, path)
    return mem


def record_query(query: str, path: Path = MEMORY_PATH) -> dict:
    """Push a query onto the bounded recent_queries deque."""
    mem = load_memory(path)
    mem["recent_queries"].append(query)
    if len(mem["recent_queries"]) > MAX_RECENT_QUERIES:
        mem["recent_queries"] = mem["recent_queries"][-MAX_RECENT_QUERIES:]
    save_memory(mem, path)
    return mem


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity of `a` against each row in `b`."""
    a_n = a / (np.linalg.norm(a) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return b_n @ a_n


def top_relevant_facts(
    query: str,
    embed_fn,
    k: int = MEMORY_TOP_K,
    path: Path = MEMORY_PATH,
) -> list[str]:
    """Return up to `k` facts most relevant to `query`.

    `embed_fn` is a callable that maps a list of strings to a numpy array of
    embeddings; injected to avoid loading the model in this module.
    """
    mem = load_memory(path)
    facts: list[str] = mem.get("facts", [])
    if not facts or k <= 0:
        return []
    if len(facts) <= k:
        return facts
    vecs = np.asarray(embed_fn(facts), dtype=np.float32)
    qv = np.asarray(embed_fn([query]), dtype=np.float32)[0]
    sims = _cosine(qv, vecs)
    order = np.argsort(-sims)[:k]
    return [facts[i] for i in order]


def format_memory_block(facts: Iterable[str]) -> str:
    """Render facts as a compact bullet list for prompt injection."""
    facts = list(facts)
    if not facts:
        return ""
    return "RELEVANT MEMORY:\n" + "\n".join(f"- {f}" for f in facts)
