"""DeepSeek LLM layer.

Builds RAG / task / calendar / meta prompts, calls the DeepSeek
OpenAI-compatible chat API (with optional streaming and JSON mode), and
runs a citation-fidelity check against the retrieved chunks.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from typing import Generator, Iterable

import httpx

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_TIMEOUT,
)

NO_MARKDOWN = (
    " IMPORTANT FORMATTING: respond in plain prose only. "
    "Do NOT use markdown — no asterisks for bold/italic, no hashes for headings, "
    "no backticks. If you need emphasis, use natural language. "
    "Lists are allowed only as plain hyphen bullets."
)

# Math typesetting directive. The frontend renders KaTeX over $...$ (inline)
# and $$...$$ (display), so when the model writes math it MUST use these
# delimiters or the UI shows ugly plain text. Strong wording because the model
# tends to drop the directive when other long instructions (graph-global,
# strict-grounding) follow it. We now place this LAST in the system prompt so
# recency weight is on our side.
MATH_TYPESETTING = (
    "MATH TYPESETTING — MANDATORY: every mathematical expression, equation, "
    "identity, or symbolic statement in your reply MUST be wrapped in LaTeX "
    "delimiters. Inline math uses $...$. Display math on its own line uses "
    "$$...$$. Examples of correct usage: $\\sin^2 x + \\cos^2 x = 1$, "
    "$\\sin(2x) = 2 \\sin x \\cos x$, $\\theta = 53.13^\\circ$, "
    "$\\tan\\theta = \\frac{4}{3}$, $x = \\frac{3}{5}$. Use \\sin, \\cos, "
    "\\tan, \\theta, \\pi, \\frac{a}{b}, x^2, \\sqrt{...}. Plain English words "
    "stay un-wrapped (e.g. 'In the first quadrant,...'), but the moment you "
    "write a variable, number-with-units in a calculation, or symbol, wrap it. "
    "Answers without delimited math will not render correctly for the user."
)

# Diagram-output directive. The model emits inline JSON specs which the backend
# converts to PNGs. Limited to single-variable functions of x; use sparingly
# (one plot per answer is usually enough) so the reply doesn't become a slideshow.
PLOT_SYNTAX = (
    "DIAGRAMS — when a graph would help the student understand the answer "
    "(e.g. they ask 'what does sin(x) look like?', or you need to show "
    "a parabola, an exponential, an integration region, a curve's behaviour), "
    "emit a plot spec INLINE in your answer using this exact format:\n\n"
    "[[PLOT:{\"expr\":\"sin(x)\",\"x_min\":-3.14159,\"x_max\":3.14159,"
    "\"title\":\"y = sin(x)\"}]]\n\n"
    "Place the marker on its own line where you want the figure to appear. "
    "Fields: expr (required, function of x using sin/cos/tan/exp/log/sqrt/abs/"
    "pi/e and operators + - * / **), x_min, x_max (numbers), title, x_label, "
    "y_label (optional strings). Use ** for powers (x**2 not x^2). The "
    "expression must depend only on the variable x. Use plots only when they "
    "genuinely add understanding — one per answer at most. If you can't "
    "express the diagram as y = f(x) (e.g. geometric figures, free-body "
    "diagrams), do not use this syntax — describe the figure in words instead."
)

SOCRATIC_PREFIX = (
    "SOCRATIC MODE IS ACTIVE. You are tutoring a student. Do not state the answer "
    "outright. Instead, ask 1–3 short, targeted questions that lead the student to "
    "discover the answer themselves. Anchor your questions in the provided context "
    "so the student knows where to look. After their reply, build on their reasoning "
    "with the next question. Only confirm a final answer if the student explicitly "
    "asks you to stop the Socratic dialogue.\n\n"
    "Adapt your questioning to the subject:\n"
    "- PHYSICS: ask what principle applies, what's given vs. unknown, what units the "
    "  answer should be in, and whether a free-body or energy approach fits. Never "
    "  give the final numerical answer first.\n"
    "- MATHEMATICS: ask which technique fits, what the next algebraic step would be, "
    "  and whether the student can sanity-check the result (units, limits, boundary "
    "  cases). Make them write the step before you confirm it.\n"
    "- ENGLISH / LITERATURE: ask what evidence in the text supports a reading, which "
    "  literary devices the author uses, and how the passage connects to themes. "
    "  Always point to a line or stanza rather than summarising.\n"
    "- PERSUASIVE WRITING: ask what the claim is, what the supporting warrant is, "
    "  whether the evidence is sufficient, who the audience is, and what the "
    "  strongest counter-argument would be. Push the student to strengthen the "
    "  argument rather than rewriting it for them. "
)

SYSTEM_BASE = (
    "You are a precise personal assistant. "
    "Answer only from the provided context. "
    "If the context doesn't contain the answer, "
    'say "I don\'t have that in my knowledge base."'
)

SYSTEM_GRAPH_LOCAL = (
    "You are a Graph-RAG assistant. The user is asking about specific entities. "
    "You are given (a) retrieved text chunks, (b) the relevant entity neighbourhood "
    "from a knowledge graph, and (c) the dominant community summary. "
    "Ground every claim in the provided material. Prefer graph relations when "
    "explaining connections. If the material doesn't contain the answer, say so."
)

SYSTEM_GRAPH_GLOBAL = (
    "You are a Graph-RAG assistant answering a corpus-wide question. "
    "You are given the top community summaries detected in the user's knowledge base. "
    "Synthesise across them: identify common themes, contrasts, and the strongest "
    "supported claims. Cite community titles inline as (community: <title>). "
    "If communities don't cover the question, say so honestly."
)

SYSTEM_TASK = (
    "You are a precise personal assistant helping with a task "
    "(writing, summarising, translating, drafting). "
    "Use the retrieved context if it is relevant; otherwise rely on general knowledge."
)

SYSTEM_CALENDAR = (
    "You are a calendar assistant. Extract date, time, duration, title and description "
    "from the user request and respond with a valid RFC 5545 .ics block wrapped in "
    "```ics fences. Use UTC if the timezone is ambiguous."
)

SYSTEM_META = (
    "You are answering a question about yourself. "
    "You are a local RAG assistant: sentence-transformers embeddings, TurboVec index, "
    "DeepSeek API for generation, optional rolling memory layer. "
    "Be brief and honest."
)

SYSTEM_STRUCTURED_SUFFIX = (
    "\n\nIMPORTANT: respond with VALID JSON only, no prose, no markdown fences. "
    'Schema: {"columns": ["..."], "rows": [["..."], ...], "notes": "optional"}'
)


@dataclass
class LLMResponse:
    """Final response from the LLM layer."""
    answer: str
    sources_used: list[str] = field(default_factory=list)
    tokens_used: int = 0
    structured: dict | None = None
    fidelity_warnings: list[str] = field(default_factory=list)
    intent: str = "knowledge"


def build_context_block(chunks: list[dict]) -> str:
    """Render retrieved chunks into the CONTEXT section of the prompt."""
    if not chunks:
        return "CONTEXT:\n(no relevant context retrieved)"
    parts = ["CONTEXT:"]
    for c in chunks:
        parts.append(f"{c['text']} — Source: {c['source']}")
    return "\n".join(parts)


def build_graph_block(entities: list[str], edges: list[str], communities: list) -> str:
    """Render entity neighbourhood + community context for graph queries.

    `communities` is a list of community.Community objects (duck-typed: any object
    with title/summary/findings attributes works).
    """
    parts: list[str] = []
    if entities:
        parts.append("RELATED ENTITIES:\n" + ", ".join(entities))
    if edges:
        parts.append("GRAPH RELATIONS:\n" + "\n".join(f"- {e}" for e in edges))
    if communities:
        block = ["COMMUNITY CONTEXT:"]
        for c in communities:
            title = getattr(c, "title", "") or f"community-{getattr(c, 'id', '?')}"
            summary = getattr(c, "summary", "")
            findings = getattr(c, "findings", []) or []
            block.append(f"• {title}")
            if summary:
                block.append(f"  {summary}")
            for f in findings[:4]:
                block.append(f"  - {f}")
        parts.append("\n".join(block))
    return "\n\n".join(parts)


def build_prompt(
    query: str,
    chunks: list[dict],
    memory_block: str,
    intent: str,
    structured: bool,
    *,
    graph_mode: str | None = None,
    graph_block: str = "",
    socratic: bool = False,
    extra_system: str = "",
    guardrail_prompt: str = "",
) -> list[dict]:
    """Assemble the OpenAI-style chat messages list.

    For knowledge intent, `graph_mode` may be "local" or "global" to swap in
    a graph-aware system prompt and inject `graph_block` (entities/edges/community).
    `extra_system` is appended to the system message — used by the Hermes
    corrector to inject strict-grounding directives when past similar queries
    have produced unsupported claims.
    `guardrail_prompt` enforces per-school curriculum scoping and Socratic
    constraints for students in a licensed school org.
    """
    if intent == "knowledge" and graph_mode == "global":
        system = SYSTEM_GRAPH_GLOBAL
    elif intent == "knowledge" and graph_mode == "local":
        system = SYSTEM_GRAPH_LOCAL
    else:
        system = {
            "knowledge": SYSTEM_BASE,
            "task": SYSTEM_TASK,
            "calendar": SYSTEM_CALENDAR,
            "meta": SYSTEM_META,
        }.get(intent, SYSTEM_BASE)

    # Guardrails go FIRST — primacy effect. The model reads system prompt
    # top-to-bottom; instructions at the top carry more weight and are harder
    # for users to override with "ignore previous instructions" attacks.
    if guardrail_prompt:
        system = guardrail_prompt + "\n\n" + system

    if structured:
        system = system + SYSTEM_STRUCTURED_SUFFIX
    else:
        system = system + NO_MARKDOWN
    if socratic and not structured:
        system = SOCRATIC_PREFIX + system
    # School guardrails activate school-level Socratic enforcement even when the
    # user hasn't toggled /socratic manually.
    if extra_system:
        system = system + "\n\n" + extra_system
    # MATH_TYPESETTING + PLOT_SYNTAX go LAST so recency weight beats Hermes'
    # strict-grounding directive and the graph-global summarisation prompt.
    # Without this order the model drops the formatting/diagram instructions
    # under longer prompts.
    if not structured:
        system = system + "\n\n" + MATH_TYPESETTING + "\n\n" + PLOT_SYNTAX

    user_parts: list[str] = []
    if memory_block:
        user_parts.append(memory_block)
    if graph_block:
        user_parts.append(graph_block)
    if intent == "knowledge":
        user_parts.append(build_context_block(chunks))
    elif chunks:
        user_parts.append("(Optional reference context follows)\n" + build_context_block(chunks))
    user_parts.append(f"USER QUESTION: {query}")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def _require_key() -> str:
    """Return the DeepSeek API key or raise a helpful error."""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Export it before running the server: "
            "`export DEEPSEEK_API_KEY=sk-...`"
        )
    return DEEPSEEK_API_KEY


def call_deepseek(
    messages: list[dict],
    stream: bool = False,
    json_mode: bool = False,
) -> dict | Generator[str, None, dict]:
    """Call DeepSeek chat completions.

    Non-streaming → returns the parsed JSON response.
    Streaming → yields content deltas (strings) and the final dict via `value`
    of `StopIteration` (use `collect_stream` for convenience).
    """
    key = _require_key()
    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/v1/chat/completions"
    payload: dict = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "stream": stream,
        "temperature": 0.2,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    if not stream:
        with httpx.Client(timeout=DEEPSEEK_TIMEOUT) as client:
            r = client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            return r.json()
    return _stream_deepseek(url, payload, headers)


def _stream_deepseek(url: str, payload: dict, headers: dict) -> Generator[str, None, dict]:
    """Stream tokens from DeepSeek and yield content deltas as they arrive."""
    final_text: list[str] = []
    tokens_used = 0
    with httpx.Client(timeout=DEEPSEEK_TIMEOUT) as client:
        with client.stream("POST", url, json=payload, headers=headers) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    final_text.append(delta)
                    yield delta
                if "usage" in chunk and chunk["usage"]:
                    tokens_used = chunk["usage"].get("total_tokens", tokens_used)
    return {"text": "".join(final_text), "tokens": tokens_used}


def collect_stream(gen: Generator[str, None, dict], echo: bool = False) -> dict:
    """Drain a streaming generator and return the final dict."""
    try:
        while True:
            piece = next(gen)
            if echo:
                sys.stdout.write(piece)
                sys.stdout.flush()
    except StopIteration as stop:
        return stop.value or {"text": "", "tokens": 0}


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{3,}")
_STOPWORDS = {
    "this", "that", "with", "from", "have", "been", "they", "their", "would",
    "should", "could", "about", "into", "your", "what", "when", "where", "which",
    "while", "there", "these", "those", "than", "then", "also", "such", "some",
    "more", "most", "many", "much", "very", "will", "just", "like", "only",
    "over", "under", "between", "after", "before", "because", "during",
}


def _content_tokens(text: str) -> set[str]:
    """Lowercased content tokens with stopwords filtered out."""
    return {w.lower() for w in _WORD_RE.findall(text or "") if w.lower() not in _STOPWORDS}


def citation_fidelity(answer: str, chunks: Iterable[dict]) -> list[str]:
    """Flag answer sentences whose content tokens aren't grounded in any chunk.

    Returns a list of human-readable warnings; an empty list means every
    sentence had at least some lexical grounding in the retrieved context.
    """
    if not answer.strip():
        return []
    chunks = list(chunks)
    if not chunks:
        return ["⚠️ No retrieved context — the entire answer is ungrounded."]
    grounded_tokens: set[str] = set()
    for c in chunks:
        grounded_tokens |= _content_tokens(c.get("text", ""))
    warnings: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", answer.strip()):
        toks = _content_tokens(sent)
        if not toks:
            continue
        overlap = toks & grounded_tokens
        # Heuristic: if fewer than 20% of content tokens land in the context,
        # treat the sentence as ungrounded.
        if len(overlap) / max(1, len(toks)) < 0.2:
            warnings.append(f"⚠️ Ungrounded claim: \"{sent.strip()}\"")
    return warnings


def maybe_parse_structured(answer: str) -> dict | None:
    """Try to parse a JSON table payload out of the answer; return None on failure."""
    text = answer.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    try:
        data = json.loads(text)
    except Exception:
        return None
    if isinstance(data, dict) and "columns" in data and "rows" in data:
        return data
    return None
