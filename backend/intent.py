"""Intent router.

Classifies an incoming query into one of:
    knowledge | task | calendar | meta

`knowledge` queries are further split into `local` (about specific entities)
vs `global` (corpus-wide themes / summarisation) via `graph_mode()`. Local
queries traverse the graph around matched entities; global queries map-reduce
over community summaries (Microsoft GraphRAG pattern).

Uses a fast keyword/regex heuristic. This keeps routing local and deterministic;
the LLM is reserved for actual generation.
"""
from __future__ import annotations

import re
from typing import Literal

Intent = Literal["knowledge", "task", "calendar", "meta"]
GraphMode = Literal["local", "global"]

_CALENDAR_PATTERNS = [
    r"\b(remind|reminder|schedule|book|set up|meeting|appointment|calendar)\b",
    r"\b(tomorrow|today|tonight|next (week|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b",
    r"\bon\s+(mon|tue|wed|thu|fri|sat|sun)",
]

_TASK_PATTERNS = [
    r"\b(write|draft|compose|reply|respond)\s+(an?\s+)?(email|message|note|reply|response)\b",
    r"\b(summari[sz]e|translate|rephrase|rewrite|paraphrase|proofread)\b",
    r"\b(generate|create)\s+(an?\s+)?(outline|plan|agenda)\b",
]

_META_PATTERNS = [
    r"\b(who are you|what can you do|how do you work)\b",
    r"\byour (capabilities|features|model|stack|memory|index|knowledge base)\b",
    r"\b(which|what) (model|llm) (are|do) you\b",
    r"\bwhat model are you (using|on)\b",
    r"\bare you (gpt|claude|deepseek|chatgpt)\b",
]


def _match_any(text: str, patterns: list[str]) -> bool:
    """True if any of the regex patterns hits."""
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def classify(query: str) -> Intent:
    """Return the best-fit intent label for `query`."""
    q = query.strip()
    if not q:
        return "knowledge"
    if _match_any(q, _META_PATTERNS):
        return "meta"
    if _match_any(q, _CALENDAR_PATTERNS):
        return "calendar"
    if _match_any(q, _TASK_PATTERNS):
        return "task"
    return "knowledge"


def wants_structured_output(query: str, triggers: tuple[str, ...]) -> bool:
    """True if the query contains any structured-output trigger phrase."""
    q = query.lower()
    return any(t in q for t in triggers)


_GLOBAL_PATTERNS = [
    r"\b(overall|across (all|the) (docs|documents|corpus|notes))\b",
    r"\b(main|key|major|recurring) (themes?|topics?|ideas?|takeaways?|trends?)\b",
    r"\b(summari[sz]e|overview of) (the|my|all) (corpus|documents|docs|notes|knowledge base)\b",
    r"\bwhat are the (most )?(important|common|frequent) \w+\b",
    r"\b(how do .* relate|how are .* connected|cluster(s|ing)?)\b",
    r"\b(big picture|high[- ]level)\b",
]


def graph_mode(query: str) -> GraphMode:
    """Decide whether a knowledge query is `global` (community-driven) or `local`.

    Global queries are answered by map-reducing over community summaries;
    local queries pull vector hits + 1-hop graph neighborhood.
    """
    return "global" if _match_any(query, _GLOBAL_PATTERNS) else "local"
