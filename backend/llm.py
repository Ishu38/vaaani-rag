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

# Diagram-output directives. The model emits inline JSON specs which the backend
# converts to PNGs. Five marker types supported — use the right one for the need.
# Place each marker on its own line where the figure should appear. One diagram
# per answer is best; never emit more than two markers in one reply.
DIAGRAM_SYNTAX = (
    "DIAGRAMS — you can embed figures INLINE in your answer using these markers. "
    "Place each marker on its own line. Use at most 2 diagrams per reply.\n\n"

    "1. FUNCTION PLOT — for curves, parabolas, trig waves, exponentials, "
    "integration regions, derivative visuals:\n"
    "[[PLOT:{\"expr\":\"sin(x)\",\"x_min\":-3.14,\"x_max\":3.14,\"title\":\"y = sin(x)\"}]]\n"
    "Fields: expr (required, function of x with sin/cos/tan/exp/log/sqrt/abs/pi/E + - * / **), "
    "x_min, x_max (numbers, default -10, 10), title, x_label, y_label (strings). "
    "Optional fill_between: [a, b] to shade area under curve. "
    "Optional expr2: second function for comparison (dashed blue line).\n\n"

    "2. CHART — for bar charts, histograms, scatter plots, box plots, pie charts:\n"
    "[[CHART:{\"type\":\"bar\",\"labels\":[\"A\",\"B\",\"C\"],\"values\":[3,7,5],\"title\":\"Results\"}]]\n"
    "[[CHART:{\"type\":\"histogram\",\"data\":[1,2,2,3,3,3,4,5],\"bins\":6}]]\n"
    "[[CHART:{\"type\":\"scatter\",\"x\":[1,2,3,4],\"y\":[2,4,6,8]}]]\n"
    "Types: bar, histogram, scatter, boxplot, pie. Fields: type (required), "
    "labels, values/data, bins (for histogram), title, x_label, y_label.\n\n"

    "3. DOT GRAPH — for flowcharts, relationship diagrams, concept maps, "
    "syntax trees, theorem proof steps, Venn-like diagrams. Use DOT language:\n"
    "[[DOT:{\"graph\":\"digraph { rankdir=TB; A [label=\\\"Start\\\"]; A -> B; B -> C [label=\\\"Condition\\\"]; }\",\"title\":\"Flowchart\"}]]\n"
    "Fields: graph (required, DOT source string — use digraph for directed, graph for undirected; "
    "nodes with [label=\"...\"]; edges with A -> B or A -- B; use \\\" inside labels). "
    "Optional title.\n\n"

    "4. CIRCUIT — for electrical circuit diagrams in physics (Current Electricity, "
    "Kirchhoff's laws, Wheatstone bridge, etc.):\n"
    "[[CIRCUIT:{\"elements\":[{\"type\":\"battery\",\"label\":\"V\",\"value\":\"12V\"},{\"type\":\"resistor\",\"label\":\"R1\",\"value\":\"10Ω\"},{\"type\":\"line\"},{\"type\":\"ground\"}],\"title\":\"Circuit\"}]]\n"
    "Element types: resistor, capacitor, inductor, battery, diode, led, ground, "
    "switch, ammeter, voltmeter, line. Each element: type (required), label, value, "
    "direction (right/left/up/down). Elements placed left-to-right by default.\n\n"

    "5. GEOMETRY — for triangles, circles, vectors, ray optics diagrams:\n"
    "[[GEOM:{\"type\":\"triangle\",\"vertices\":[[0,0],[4,0],[2,3]],\"labels\":[\"A\",\"B\",\"C\"],\"show_angles\":true,\"right_angle\":false}]]\n"
    "[[GEOM:{\"type\":\"circle\",\"center\":[0,0],\"radius\":5,\"show_axes\":true}]]\n"
    "[[GEOM:{\"type\":\"vectors\",\"vectors\":[[0,0,3,2],[0,0,1,4]]}]]\n"
    "[[GEOM:{\"type\":\"ray_optics\",\"kind\":\"convex_lens\"}]]\n"
    "Types: triangle (with vertices, labels, show_angles, right_angle), "
    "circle (center, radius, show_axes), vectors (list of [x,y,dx,dy]), "
    "ray_optics (kind: convex_lens or concave_lens).\n\n"

    "CHOOSE THE RIGHT MARKER: "
    "- Any y=f(x) curve → PLOT "
    "- Data, distributions, comparisons → CHART "
    "- Flowcharts, trees, concept maps, syntax → DOT "
    "- Electrical circuits → CIRCUIT "
    "- Geometric shapes, vectors, optics → GEOM "
    "If a diagram doesn't fit any type, describe it in words instead."
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

SYSTEM_COT = (
    "\n\nREASONING REQUIREMENT: think step by step. Before presenting any "
    "answer, break the problem down into logical stages. Number your steps. "
    "Show the reasoning chain explicitly — derive formulas before plugging in "
    "numbers, state assumptions, and verify units at each stage. "
    "For problems: list what is given and what is to find first. "
    "For explanations: build from first principles, not from the conclusion. "
    "End with a clearly marked final answer or summary."
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

    # Direct-mode (non-Socratic) knowledge queries: enforce chain-of-thought
    # step-by-step reasoning. Socratic mode asks questions instead, and
    # structured/JSON mode already has a rigid output format.
    if intent == "knowledge" and not socratic and not structured:
        system = system + SYSTEM_COT

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
        system = system + "\n\n" + MATH_TYPESETTING + "\n\n" + DIAGRAM_SYNTAX

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
