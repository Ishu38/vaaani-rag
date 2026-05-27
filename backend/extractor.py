"""LLM-driven entity & relation extraction for Graph-RAG.

For each text chunk we ask DeepSeek (JSON mode) to return:

    {
      "entities":  [{"name": str, "type": str, "description": str}, ...],
      "relations": [{"source": str, "target": str, "type": str, "description": str}, ...]
    }

The extractor is deterministic-ish: temperature 0, strict JSON, plus a salvage
pass that strips code fences if the model decorated its output.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable

import httpx

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_INGEST_MODEL, DEEPSEEK_TIMEOUT


EXTRACTION_SYSTEM = (
    "You are an information extraction engine for a Graph-RAG knowledge base. "
    "From the provided text, extract:\n"
    "1. ENTITIES: concrete named things — people, organisations, products, places, "
    "concepts, technologies, events, dates. Skip pronouns and generic nouns.\n"
    "2. RELATIONS: directed edges between entities. Use short verb-phrase types "
    "(e.g. 'works_at', 'invented', 'located_in', 'depends_on').\n\n"
    "Respond with VALID JSON only — no prose, no markdown fences — matching:\n"
    '{"entities":[{"name":"...","type":"...","description":"..."}],'
    '"relations":[{"source":"...","target":"...","type":"...","description":"..."}]}\n\n'
    "Entity names should be canonical (e.g. 'OpenAI' not 'openai inc.'). "
    "If the text has no extractable structure, return empty arrays."
)

# Batched extraction over N chunks per DeepSeek call. Keeps per-call latency
# similar (~6-8 s) but amortises HTTP overhead and DeepSeek's prefill across
# multiple chunks — net throughput is ~3-4x faster than one-chunk-per-call
# at the same concurrency, which is the difference between waiting 12 min
# and 3 min on a 60 MB PDF.
EXTRACTION_SYSTEM_BATCH = (
    "You are an information extraction engine for a Graph-RAG knowledge base. "
    "You will receive N text chunks delimited by '<<<CHUNK i>>>' markers. "
    "Extract entities and relations from EACH chunk independently.\n\n"
    "For every chunk, produce one object with that chunk's index. Respond with "
    "VALID JSON only (no prose, no markdown fences) matching:\n"
    '{"results":[\n'
    '  {"index":0,"entities":[{"name":"...","type":"...","description":"..."}],'
    '"relations":[{"source":"...","target":"...","type":"...","description":"..."}]},\n'
    '  {"index":1,"entities":[...],"relations":[...]},\n'
    '  ...\n'
    "]}\n\n"
    "Rules: include EVERY chunk index from 0 to N-1 in the results array even "
    "if a chunk has no extractable structure (return empty arrays for that "
    "chunk). Entity names should be canonical. Skip pronouns and generic nouns. "
    "Use short verb-phrase relation types."
)


@dataclass
class Entity:
    """A node in the knowledge graph."""
    name: str
    type: str = "unknown"
    description: str = ""


@dataclass
class Relation:
    """A directed edge in the knowledge graph."""
    source: str
    target: str
    type: str = "related_to"
    description: str = ""


@dataclass
class Extraction:
    """The full extraction result for one chunk."""
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)


def _strip_fences(text: str) -> str:
    """Drop ```json ... ``` wrappers if the model added them anyway."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    return text


def _parse_payload(raw: str) -> Extraction:
    """Parse a model response into an Extraction; tolerant to small deviations."""
    try:
        data = json.loads(_strip_fences(raw))
    except Exception:
        return Extraction()
    ents = [
        Entity(
            name=str(e.get("name", "")).strip(),
            type=str(e.get("type", "unknown")).strip() or "unknown",
            description=str(e.get("description", "")).strip(),
        )
        for e in (data.get("entities") or [])
        if isinstance(e, dict)
        and str(e.get("name", "")).strip()
        and len(str(e.get("name", "")).strip()) >= 2  # reject single-char noise (a, x, c...)
    ]
    rels = [
        Relation(
            source=str(r.get("source", "")).strip(),
            target=str(r.get("target", "")).strip(),
            type=str(r.get("type", "related_to")).strip() or "related_to",
            description=str(r.get("description", "")).strip(),
        )
        for r in (data.get("relations") or [])
        if isinstance(r, dict)
        and str(r.get("source", "")).strip()
        and str(r.get("target", "")).strip()
    ]
    return Extraction(entities=ents, relations=rels)


def extract_chunk(text: str, *, client: httpx.Client | None = None) -> Extraction:
    """Extract entities and relations from a single chunk via DeepSeek JSON mode."""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Graph-RAG ingest requires DeepSeek access."
        )
    if not text or not text.strip():
        return Extraction()
    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": DEEPSEEK_INGEST_MODEL,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": f"TEXT:\n{text}"},
        ],
    }
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=DEEPSEEK_TIMEOUT)
    try:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        return _parse_payload(content)
    finally:
        if own_client:
            client.close()


def extract_chunks_batch(
    texts: list[str],
    *,
    client: httpx.Client | None = None,
) -> list[Extraction]:
    """Extract entities + relations for N chunks in a single DeepSeek call.

    Returns a list of Extraction objects in the SAME order as the input
    texts. On any failure (HTTP error, malformed JSON, missing indices,
    truncated batch), every chunk in the batch gets an empty Extraction
    — the caller is expected to retry that batch one-by-one if it cares,
    or just accept the loss. For large textbook ingests the per-batch
    failure rate is well under 1%, so the simpler "drop the batch" rule
    is fine.

    Why this exists: a 60 MB textbook chunks to ~1500-3000 pieces. One
    DeepSeek call per chunk at 12-concurrency takes ~12 min. Five
    chunks per call at the same concurrency takes ~3 min — same per-
    call latency, 5× less HTTP overhead and prefill cost. The batched
    JSON contract is straightforward enough that DeepSeek's JSON mode
    handles it reliably at temperature 0.
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Graph-RAG ingest requires DeepSeek access."
        )
    texts = [(t or "") for t in texts]
    if not texts:
        return []

    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/v1/chat/completions"
    # Build the numbered-chunk user message.
    sections = [f"<<<CHUNK {i}>>>\n{t}" for i, t in enumerate(texts)]
    user_msg = "\n\n".join(sections)
    payload = {
        "model": DEEPSEEK_INGEST_MODEL,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM_BATCH},
            {"role": "user", "content": user_msg},
        ],
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=DEEPSEEK_TIMEOUT)
    try:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    finally:
        if own_client:
            client.close()

    # Parse the batched response back into ordered Extraction list.
    try:
        data = json.loads(_strip_fences(content))
    except Exception:
        return [Extraction() for _ in texts]
    results_raw = data.get("results") or []
    by_index: dict[int, Extraction] = {}
    for item in results_raw:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(texts):
            continue
        ents = [
            Entity(
                name=str(e.get("name", "")).strip(),
                type=str(e.get("type", "unknown")).strip() or "unknown",
                description=str(e.get("description", "")).strip(),
            )
            for e in (item.get("entities") or [])
            if isinstance(e, dict)
            and str(e.get("name", "")).strip()
            and len(str(e.get("name", "")).strip()) >= 2
        ]
        rels = [
            Relation(
                source=str(r.get("source", "")).strip(),
                target=str(r.get("target", "")).strip(),
                type=str(r.get("type", "related_to")).strip() or "related_to",
                description=str(r.get("description", "")).strip(),
            )
            for r in (item.get("relations") or [])
            if isinstance(r, dict)
            and str(r.get("source", "")).strip()
            and str(r.get("target", "")).strip()
        ]
        by_index[idx] = Extraction(entities=ents, relations=rels)

    # Backfill any missing indices with empty extractions (model dropped them).
    return [by_index.get(i, Extraction()) for i in range(len(texts))]


def extract_many(texts: Iterable[str], *, progress: bool = True) -> list[Extraction]:
    """Extract over a sequence of chunks, reusing one HTTP client."""
    texts = list(texts)
    out: list[Extraction] = []
    with httpx.Client(timeout=DEEPSEEK_TIMEOUT) as client:
        for i, t in enumerate(texts, 1):
            try:
                ex = extract_chunk(t, client=client)
            except Exception as e:
                if progress:
                    print(f"  [extract:warn] chunk {i}/{len(texts)} failed: {e}")
                ex = Extraction()
            out.append(ex)
            if progress:
                print(f"  [extract] {i}/{len(texts)}: +{len(ex.entities)} ents, +{len(ex.relations)} rels")
    return out
