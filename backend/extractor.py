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
        if isinstance(e, dict) and str(e.get("name", "")).strip()
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
