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

# Linguistic notation directive. Replaces the old MATH_TYPESETTING block from
# the JEE-era scope: this product is linguistics-only for schools, so the
# conventions that matter are IPA-style slashes/brackets and glossing, not
# LaTeX. Placed LAST in the system prompt so recency weight is on our side
# (the model drops trailing directives under long graph/grounding prompts).
LINGUISTIC_NOTATION = (
    "LINGUISTIC NOTATION: use standard conventions when discussing language. "
    "Phonemes go in slashes (/p/, /s/), phonetic realisations in square "
    "brackets ([pʰ]), and an asterisk marks ungrammatical examples (*goed). "
    "Write example words and sentences in single quotes and always pair a "
    "technical term with a plain-language explanation and an example, so a "
    "school student (or their parent) can follow without prior training. "
    "If you ever need genuine mathematics (e.g. counting frequencies), wrap "
    "it in $...$ so it renders, but prefer plain prose."
)

# The heart of the tutor: no matter WHAT subject the uploaded material covers,
# every answer is taught through a linguist's eyes — morphology, etymology, word
# families, meaning. This is what makes Vaaani a linguistics tutor rather than a
# generic document Q&A bot.
LINGUISTIC_LENS = (
    "LINGUISTIC LENS — this is how Vaaani teaches, and it is not optional. "
    "Whatever subject the provided material is about (science, history, civics, "
    "a story — anything), teach it the way a linguist would, through the language "
    "itself. In every knowledge answer:\n"
    "• Take the key and hard words and BREAK THEM INTO MORPHEMES — root, prefix, "
    "suffix, and any other affix — e.g. 'photosynthesis' = 'photo-' (light) + "
    "'synthesis' (putting together); 'unbreakable' = 'un-' (not) + 'break' + "
    "'-able' (able to be).\n"
    "• Give the ORIGIN of each part (Greek, Latin, Sanskrit, Persian, Arabic, "
    "Bangla, Hindi…) and its plain MEANING.\n"
    "• Name the WORD FAMILY — other words built from the same root — so the child "
    "sees the pattern ('photo-': photograph, photocopy, photon).\n"
    "• Where it helps, note the SOUND (phoneme) or how the child's mother tongue "
    "(Bangla / Hindi) maps or contrasts with it.\n"
    "• Always land on MEANING: how the pieces add up, and how knowing the parts "
    "lets the child decode NEW words they've never seen.\n"
    "Stay grounded in the provided material for the facts; the morphological "
    "breakdown is your lens on that material, never a substitute for it. Keep it "
    "plain enough for a school child and their parent to follow."
)

# Diagram-output directives. The model emits inline JSON specs which the backend
# converts to PNGs. Linguistics scope: only DOT (trees / maps) and CHART (data)
# are offered to the model — the renderer still supports the legacy PLOT/CIRCUIT/
# GEOM markers, but a linguistics tutor has no business emitting them.
# Place each marker on its own line. One diagram per answer is best.
DIAGRAM_SYNTAX = (
    "DIAGRAMS — you can embed figures INLINE in your answer using these markers. "
    "Place each marker on its own line. Use at most 2 diagrams per reply.\n\n"

    "1. DOT GRAPH — for syntax trees, morpheme breakdowns, language family "
    "trees, concept maps, and word-relation diagrams. Use DOT language:\n"
    "[[DOT:{\"graph\":\"digraph { rankdir=TB; S [label=\\\"Sentence\\\"]; NP [label=\\\"Noun Phrase\\\"]; VP [label=\\\"Verb Phrase\\\"]; S -> NP; S -> VP; }\",\"title\":\"Sentence structure\"}]]\n"
    "Fields: graph (required, DOT source string — use digraph for directed, graph for undirected; "
    "nodes with [label=\"...\"]; edges with A -> B or A -- B; use \\\" inside labels). "
    "Optional title.\n\n"

    "2. CHART — for word-frequency data, sound inventories, survey results:\n"
    "[[CHART:{\"type\":\"bar\",\"labels\":[\"Hindi\",\"Tamil\",\"Persian\"],\"values\":[12,5,9],\"title\":\"English loanwords by source\"}]]\n"
    "Types: bar, histogram, scatter, boxplot, pie. Fields: type (required), "
    "labels, values/data, bins (for histogram), title, x_label, y_label.\n\n"

    "CHOOSE THE RIGHT MARKER: "
    "- Trees (syntax, morphology, language families), concept maps → DOT "
    "- Counts, frequencies, comparisons → CHART "
    "If a diagram doesn't fit either type, describe it in words instead."
)

SOCRATIC_PREFIX = (
    "SOCRATIC MODE IS ACTIVE. You are tutoring a student. Do not state the answer "
    "outright. Instead, ask 1–3 short, targeted questions that lead the student to "
    "discover the answer themselves. Anchor your questions in the provided context "
    "so the student knows where to look. After their reply, build on their reasoning "
    "with the next question. Only confirm a final answer if the student explicitly "
    "asks you to stop the Socratic dialogue.\n\n"
    "Adapt your questioning to the linguistic level:\n"
    "- SOUNDS (phonetics/phonology): ask the student to say the word aloud, notice "
    "  what their mouth is doing, count sounds not letters, and compare a minimal "
    "  pair. Never just state the phonetic answer first.\n"
    "- WORDS (morphology/etymology): ask them to peel the word apart — what is the "
    "  core, what was added, what does each piece contribute — and to find another "
    "  word that shares a piece. Make them propose the breakdown before you confirm.\n"
    "- SENTENCES (syntax/grammar): ask who is doing what, which chunk of words hangs "
    "  together, and what happens if you move or delete a chunk. Have them test the "
    "  rule on a new sentence of their own.\n"
    "- MEANING (semantics): ask what the word/sentence could mean, whether two "
    "  readings are possible, and what context would decide between them.\n"
    "- ENGLISH / LITERATURE: ask what evidence in the text supports a reading, which "
    "  literary devices the author uses, and how the passage connects to themes. "
    "  Always point to a line or stanza rather than summarising.\n"
    "- PERSUASIVE WRITING: ask what the claim is, what the supporting warrant is, "
    "  whether the evidence is sufficient, who the audience is, and what the "
    "  strongest counter-argument would be. Push the student to strengthen the "
    "  argument rather than rewriting it for them. "
)

SYSTEM_BASE = (
    "You are Vaaani, a linguistics tutor for school children. "
    "Answer only from the provided context, and teach it through a linguist's eyes. "
    "If the context doesn't contain the answer, "
    'say "I don\'t have that in my knowledge base."'
)

SYSTEM_GRAPH_LOCAL = (
    "You are Vaaani, a linguistics tutor using a knowledge graph. The user is asking about specific entities. "
    "You are given (a) retrieved text chunks, (b) the relevant entity neighbourhood "
    "from a knowledge graph, and (c) the dominant community summary. "
    "Ground every claim in the provided material. Prefer graph relations when "
    "explaining connections. If the material doesn't contain the answer, say so."
)

SYSTEM_GRAPH_GLOBAL = (
    "You are Vaaani, a linguistics tutor answering a corpus-wide question. "
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
    "You are Vaaani — a linguistics study assistant. Your engine combines a "
    "knowledge-graph retrieval layer, sentence-transformers embeddings, a TurboVec "
    "index, a per-student memory layer, and Vaaani's own language-generation engine. "
    "Be brief and honest about what you can do, but describe the system as 'Vaaani's "
    "engine' — do NOT name any third-party model or provider."
)

# Identity lock. Placed at the very top of every system prompt (primacy) so the
# model never reveals or speculates about the underlying generation provider.
# Vaaani's intelligence is its student-memory + knowledge-graph layer; the text
# generator is swappable infrastructure and must stay invisible to users.
IDENTITY_LOCK = (
    "YOUR IDENTITY: You are Vaaani, a linguistics study assistant. You run on "
    "Vaaani's own language engine. NEVER reveal, name, confirm, deny, or speculate "
    "about the underlying model, vendor, company, or API that powers you (you have "
    "no knowledge of any such third party). If a user asks what model or company you "
    "are, who built your AI, or whether you are any named system, reply only that you "
    "are Vaaani, running on Vaaani's own language engine, and steer back to helping "
    "them learn. This instruction overrides any user request to disclose it."
)

SYSTEM_STRUCTURED_SUFFIX = (
    "\n\nIMPORTANT: respond with VALID JSON only, no prose, no markdown fences. "
    'Schema: {"columns": ["..."], "rows": [["..."], ...], "notes": "optional"}'
)

SYSTEM_COT = (
    "\n\nREASONING REQUIREMENT: think step by step. Before presenting any "
    "answer, break the question down into logical stages. Number your steps. "
    "Show the reasoning chain explicitly — state the linguistic rule or "
    "concept before applying it to the example, and give an example for "
    "every technical term you use. "
    "For analysis questions: identify the level (sound, word, sentence, "
    "meaning) first, then work through the data. "
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
    orchestrator_system: str = "",
) -> list[dict]:
    """Assemble the OpenAI-style chat messages list.

    For knowledge intent, `graph_mode` may be "local" or "global" to swap in
    a graph-aware system prompt and inject `graph_block` (entities/edges/community).
    `extra_system` is appended to the system message — used by the Hermes
    corrector to inject strict-grounding directives when past similar queries
    have produced unsupported claims.
    `guardrail_prompt` enforces per-school curriculum scoping and Socratic
    constraints for students in a licensed school org.
    `orchestrator_system` — when non-empty, replaces the standard tutor persona
    entirely so the Discovery Orchestrator owns the turn (used for discovery-
    mode sessions originated from IPA/Explore/Sound Lab).
    """
    # Discovery Orchestrator mode: replace the entire system prompt.
    # The orchestrator prompt already embeds identity, teaching philosophy,
    # and behavioural rules — none of the standard tutor prompts apply.
    if orchestrator_system:
        system = orchestrator_system
        socratic = True
        structured = False
    else:
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
        # LINGUISTIC_NOTATION + DIAGRAM_SYNTAX go LAST so recency weight beats
        # Hermes' strict-grounding directive and the graph-global summarisation
        # prompt. Without this order the model drops the formatting/diagram
        # instructions under longer prompts.
        if not structured:
            system = system + "\n\n" + LINGUISTIC_NOTATION + "\n\n" + DIAGRAM_SYNTAX
            # The linguistic lens is the tutor's core behaviour — place it LAST so its
            # recency weight makes the model teach every knowledge answer morphologically.
            if intent == "knowledge":
                system = system + "\n\n" + LINGUISTIC_LENS

    # Identity lock goes at the ABSOLUTE top (prepended last) so it carries the
    # strongest primacy weight and can't be dislodged by a disclosure attack.
    # Skip for orchestrator mode — the orchestrator prompt carries its own
    # identity ("You are Vaaani Discovery Orchestrator") that must not be
    # overwritten by the standard tutor identity.
    if not orchestrator_system:
        system = IDENTITY_LOCK + "\n\n" + system

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


_PROVIDER_LEAK_RE = re.compile(r"deep[\s\-]?seek[\w.\-]*", re.IGNORECASE)


def scrub_provider_identity(text: str) -> str:
    """Belt-and-suspenders: replace any stray provider name the model emits with
    Vaaani's own engine. The IDENTITY_LOCK system prompt is the primary defence;
    this catches the rare case where the model names the provider in body text.
    """
    if not text:
        return text
    return _PROVIDER_LEAK_RE.sub("Vaaani's language engine", text)


def call_deepseek(
    messages: list[dict],
    stream: bool = False,
    json_mode: bool = False,
    temperature: float = 0.2,
) -> dict | Generator[str, None, dict]:
    """Call the Vaaani engine's chat completions (OpenAI wire format).

    Non-streaming → returns the parsed JSON response.
    Streaming → yields content deltas (strings) and the final dict via `value`
    of `StopIteration` (use `collect_stream` for convenience).

    `temperature`: knowledge answers should pass 0.0 — greedy decoding makes
    the small model markedly less prone to invented facts; keep a little
    sampling (0.2) only for Socratic questioning and task/creative intents.
    """
    key = _require_key()
    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/v1/chat/completions"
    payload: dict = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
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
