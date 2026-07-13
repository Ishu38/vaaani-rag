#!/usr/bin/env python3
"""Phase 2 — Graph-First Query Router.

Routes educational queries through the structural linguistics knowledge
graph BEFORE any vector search or LLM call.  When the graph has a
deterministic answer (confidence >= 3), returns it immediately — zero
LLM tokens, zero hallucination risk, sub-50ms response time.

Confidence scale:
  5  — perfect match: all hops found, rich context
  4  — strong match: root + meaning + siblings
  3  — good match: root + meaning or meaning + siblings
  2  — partial match: found the word but incomplete traversal
  1  — weak match: fuzzy hit, uncertain
  0  — no match: fall through to vector search + LLM

Intents (each maps to a traversal pattern):
  morphology  — "what does X mean", "break down X", "root of X"
  phonics     — "how is X written", "what sound", "spelling of"
  semantics    — "what does X mean", "define X", "meaning of"
  discovery    — "what next", "what to learn after", "suggest"
  etymology    — "where does X come from", "origin of"
  family       — "words like X", "same root", "family of"
  comparison   — "X vs Y", "compare X and Y"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from graph import KnowledgeGraph, normalize


# ── Intent classifier (regex — zero LLM) ────────────────────────────────────

_INTENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("morphology", re.compile(
        r"(?i)\b(break\s?(down|apart)|what\s+(is|are)\s+the\s+(root|piece|part)s?\s+(of|in)"
        r"|root\s+(of|word)\b|prefix\s+(of|in)\b|suffix\s+(of|in)\b"
        r"|part of (the )?word|how is .+ (built|made|formed)|morpheme)"),
    ),
    ("phonics", re.compile(
        r"(?i)\b(what\s+sound|how\s+(is|to|do\s+i|do\s+you)\s+(say|spell|pronounce|write)|"
        r"spelling\s+(of|pattern)|written\s+(as|like)|pronunciation|"
        r"how\s+(is|does).+\bsound\b|\bphoneme|grapheme|digraph|how.*written|how.*spelled|"
        r"sound\s+like\b|\bsounds?\s+like\b)"),
    ),
    ("semantics", re.compile(
        r"(?i)\b(what\s+(does|is)\s+(the\s+)?meaning\s+(of)?|"
        r"define|definition\s+(of)?|what.*(mean|means)\b|"
        r"gloss\s+(of|for))"),
    ),
    ("etymology", re.compile(
        r"(?i)\b(where\s+(does|is).+come\s+from|origin( of)?|"
        r"etymology|language\s+of|cognate|from\s+(which|what)\s+language|"
        r"sanskrit|hindi|bengali|latin|greek)\b"),
    ),
    ("discovery", re.compile(
        r"(?i)\b(what\s+(next|should\s+(i|I)\s+learn|can\s+(i|I)\s+(learn|discover)|"
        r"comes\s+(after|next)|do\s+(i|I)\s+learn\s+(next|after))|"
        r"suggest\s+(a\s+)?(word|concept)|next\s+(word|concept|lesson)|"
        r"what\s+to\s+(learn|discover|study)|recommend)"),
    ),
    ("family", re.compile(
        r"(?i)\b(words\s+(like|with(\s+the)?\s+(same\s+)?root|in\s+the\s+(same\s+)?family)"
        r"|similar\s+(to|words)|family\s+of|siblings?\s+of|"
        r"same\s+(root|piece)\s+as|related\s+words|"
        r"what\s+(other|else)\s+(words\s+)?(have|use|contain|share)|"
        r"what\s+family\b|belongs?\s+to\b|belong\s+to\b)"),
    ),
    ("comparison", re.compile(
        r"(?i)\b(compare|vs\.?|versus|difference\s+between|"
        r"what'?s?\s+the\s+difference|how\s+(is|are).+different\s+(from|than))"),
    ),
    ("readiness", re.compile(
        r"(?i)\b(am\s+(i|I)\s+ready\s+(for|to\s+learn)|can\s+(i|I)\s+learn|"
        r"prerequisites?\s+(for|of|to)|what\s+do\s+(i|I)\s+need\s+(before|to\s+know)"
        r"|do\s+(i|I)\s+know\s+enough\s+(for|to)|"
        r"what\s+(is|are)\s+the\s+prerequisites?\s+(for|of|to)|"
        r"how\s+can\s+(i|I)\s+(learn|master|understand))"),
    ),
]


def classify_educational_intent(query: str) -> str:
    """Return the most specific educational intent, or 'knowledge' (fallback)."""
    scored: list[tuple[int, str]] = []
    for name, pat in _INTENT_PATTERNS:
        matches = pat.findall(query)
        if matches:
            # Longer match patterns = more specific = higher priority
            total = sum(len("".join(m)) if isinstance(m, tuple) else len(str(m))
                       for m in matches)
            scored.append((total, name))
    if scored:
        scored.sort(reverse=True)
        return scored[0][1]
    return "knowledge"


# ── Fuzzy node finder ───────────────────────────────────────────────────────

def _singularize(word: str) -> str:
    """Crude singularization for common patterns (child -> child, fishes -> fish)."""
    w = word.lower().strip()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("sses"):  # glasses → glass
        return w[:-2]
    if w.endswith("shes") or w.endswith("ches") or w.endswith("xes"):
        return w[:-2]
    if w.endswith("ses") or w.endswith("zes"):
        return w[:-1]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def _fuzzy_find(kg: KnowledgeGraph, word: str) -> Optional[str]:
    """Find the best matching node key for a user-supplied word.
    
    Tries in order: exact match, singular/plural variants, full query match,
    display-name substring, prefix match.
    """
    q = normalize(word)
    if not q:
        return None
    if kg.g.has_node(q):
        return q

    # Try without trailing 's' (binoculars → binocular)
    singular = _singularize(q)
    if singular != q and kg.g.has_node(singular):
        return singular

    # Try with added 's' (tripod → tripods)
    plural = q + "s"
    if plural != q and kg.g.has_node(plural):
        return plural

    # Scan all node display names for a match
    q_len = len(q)
    for nid, data in kg.g.nodes(data=True):
        disp = normalize(data.get("display", ""))
        if not disp:
            continue
        # Full display match
        if disp == q:
            return nid
        # Display contains query (e.g. query="graph" matches "photograph")
        if len(disp) > 4 and q_len >= 3 and q in disp:
            return nid
        # Query contains display (e.g. query="binoculars" contains "bi")
        if len(q) > 4 and len(disp) >= 3 and disp in q:
            return nid

    # Prefix match (first 3+ letters)
    if q_len >= 3:
        for nid, data in kg.g.nodes(data=True):
            disp = normalize(data.get("display", ""))
            if disp.startswith(q):
                return nid

    return None


def _find_words_in_query(kg: KnowledgeGraph, query: str) -> list[str]:
    """Extract candidate words from the query that exist in the graph.
    Returns word and root nodes first (most relevant), then others."""
    tokens = re.findall(r"[a-z]{3,}", query.lower())
    word_roots: list[str] = []
    others: list[str] = []
    for t in tokens:
        nid = _fuzzy_find(kg, t)
        if nid:
            ntype = kg.g.nodes[nid].get("type", "")
            if ntype in ("word", "root"):
                word_roots.append(nid)
            else:
                others.append(nid)
    return word_roots + others


# ── Traversal patterns per intent ───────────────────────────────────────────

@dataclass
class GraphResult:
    answer: str
    confidence: int  # 0–5
    intent: str
    entities: list[str] = field(default_factory=list)
    traceback: str = ""  # Reversal 3 — exactly which graph path produced this answer


_NO_ANSWER = GraphResult("", 0, "knowledge")


def _graph_route_morphology(kg: KnowledgeGraph, word: str) -> GraphResult:
    """word → root → meaning + siblings."""
    nid = _fuzzy_find(kg, word)
    if not nid:
        return _NO_ANSWER
    node = kg.g.nodes[nid]
    display = node.get("display", nid)
    parts: list[str] = []

    # Find root via outgoing root_of edges (from root to word) — we need incoming
    root_keys: list[str] = []
    root_display = ""
    meaning = ""
    for src, _, data in kg.g.in_edges(nid, data=True):
        if data.get("type") == "root_of":
            root_keys.append(src)
            rn = kg.g.nodes.get(src, {})
            root_display = rn.get("display", src)
            # Find meaning of this root
            for _, mdst, md in kg.g.edges(src, data=True):
                if md.get("type") == "means":
                    meaning = kg.g.nodes.get(mdst, {}).get("display", mdst)
                    break

    if not root_keys:
        return GraphResult(
            f"I found {display} in the graph but couldn't find its root. "
            f"Can you try asking differently?",
            1, "morphology", [nid])

    # Find siblings (other words with same root)
    siblings: list[str] = []
    for rk in root_keys:
        for _, dst, d in kg.g.edges(rk, data=True):
            if d.get("type") in ("root_of", "used_in") and dst != nid:
                sd = kg.g.nodes.get(dst, {}).get("display", dst)
                if sd not in siblings:
                    siblings.append(sd)

    # Build answer
    parts.append(f"{display} comes from the root {root_display}")
    if meaning:
        parts.append(f" which means \u201c{meaning}\u201d")
    parts.append(".")

    if siblings:
        parts.append(f" Words in the same family: ")
        parts.append(", ".join(siblings[:6]))
        parts.append(".")

    parts.append(f" Every word that has {root_display} inside it carries the idea of \u201c{meaning or ''}\u201d.")

    conf = 5 if meaning and siblings else (4 if meaning else 3)
    return GraphResult("".join(parts), conf, "morphology", [nid] + root_keys + siblings)


def _graph_route_phonics(kg: KnowledgeGraph, query: str) -> GraphResult:
    """Alias → phoneme → graphemes + example words, OR word → phoneme → graphemes."""
    # Try: find alias node (Fan Sound, Snake Sound...)
    # First, look for "the X sound" pattern — prefer explicit "the" prefix
    sound_m = re.search(r"(?i)\bthe\s+(\w+(?:\s+\w+)?)\s+sound\b", query)
    if not sound_m:
        # Fall back to bare "X sound" pattern
        sound_m = re.search(r"(?i)\b(\w{4,}(?:\s+\w+)?)\s+sound\b", query)
    alias_candidate = ""
    if sound_m:
        alias_candidate = sound_m.group(1).strip().lower()

    # Look for words in the query that map to phonemes
    words = _find_words_in_query(kg, query)
    phoneme_key = None
    found_display = ""

    # Try alias match first
    if alias_candidate:
        # Try exact match as alias node
        ak = _fuzzy_find(kg, alias_candidate)
        if ak and kg.g.nodes.get(ak, {}).get("type") == "alias":
            for _, dst, d in kg.g.edges(ak, data=True):
                if d.get("type") == "alias_of":
                    phoneme_key = dst
                    found_display = kg.g.nodes.get(ak, {}).get("display", alias_candidate)
                    break
        # Try: word node → sounds_like → phoneme (for words like "fan"→/f/)
        if not phoneme_key:
            wk = _fuzzy_find(kg, alias_candidate)
            if wk:
                for _, dst, d in kg.g.edges(wk, data=True):
                    if d.get("type") == "sounds_like":
                        phoneme_key = dst
                        found_display = kg.g.nodes.get(wk, {}).get("display", alias_candidate)
                        break
        # Fallback: map word's first letter to phoneme (fan → f phoneme)
        if not phoneme_key:
            first = alias_candidate[0] if alias_candidate else ""
            pk = f"phoneme-{first}"
            if kg.g.has_node(pk):
                phoneme_key = pk
                found_display = f"the {alias_candidate} sound"

    # Try: word → sounds_like → phoneme
    if not phoneme_key and words:
        for w in words:
            for _, dst, d in kg.g.edges(w, data=True):
                if d.get("type") == "sounds_like":
                    phoneme_key = dst
                    found_display = kg.g.nodes.get(w, {}).get("display", w)
                    break
            if phoneme_key:
                break

    # Fallback: direct phoneme key lookup (e.g. "f" → phoneme-f, or "f sound")
    if not phoneme_key:
        for token in re.findall(r"[a-z]{1,4}", query.lower()):
            pk = f"phoneme-{token}"
            if kg.g.has_node(pk):
                phoneme_key = pk
                found_display = kg.g.nodes[pk].get("display", token)
                break
            # Also try: "fan" → first letter "f"
            if len(token) >= 3:
                pk = f"phoneme-{token[0]}"
                if kg.g.has_node(pk):
                    phoneme_key = pk
                    found_display = f"the {token} sound"
                    break

    if not phoneme_key:
        return _NO_ANSWER

    # Get graphemes (phoneme → written_as → grapheme)
    graphemes: list[str] = []
    for _, dst, d in kg.g.edges(phoneme_key, data=True):
        if d.get("type") == "written_as":
            gd = kg.g.nodes.get(dst, {}).get("display", dst)
            graphemes.append(gd)

    # Get example words from phoneme node's data
    pnode = kg.g.nodes[phoneme_key]
    examples = pnode.get("examples", [])

    lines: list[str] = []
    intro = found_display
    if intro.startswith("the "):
        lines.append(f"{intro[0].upper()}{intro[1:]}")
    else:
        lines.append(f"The {intro}")
    lines.append(f" is a {'voiced' if pnode.get('voice') else 'voiceless'} "
                 f"{pnode.get('place', '').lower()} {pnode.get('manner', '').lower()}")
    lines.append(". ")

    if graphemes:
        lines.append(f"It can be written as: {', '.join(graphemes)}. ")
    if examples:
        lines.append(f"Practice words: {', '.join(examples[:6])}.")

    conf = 4 if graphemes and examples else 3
    trace = _build_traceback(
        target=found_display,
        phoneme=pnode.get("display", ""),
        graphemes=graphemes,
        source="graph traversal (phonics)",
    )
    return GraphResult("".join(lines), conf, "phonics",
                        [phoneme_key] + graphemes + examples, traceback=trace)


def _graph_route_semantics(kg: KnowledgeGraph, word: str) -> GraphResult:
    """word → meaning/gloss from node descriptions or root."""
    nid = _fuzzy_find(kg, word)
    if not nid:
        return _NO_ANSWER
    node = kg.g.nodes[nid]
    display = node.get("display", nid)
    descs = node.get("descriptions", [])
    gloss = descs[0] if descs else ""
    ntype = node.get("type", "")

    if ntype == "root":
        # Root → meaning
        for _, dst, d in kg.g.edges(nid, data=True):
            if d.get("type") == "means":
                meaning = kg.g.nodes.get(dst, {}).get("display", dst)
                return GraphResult(
                    f"{display} means \u201c{meaning}\u201d. "
                    f"It is a {node.get('type','?')} used to build English words.",
                    5, "semantics", [nid, dst])

    if gloss:
        return GraphResult(
            f"{display}: {gloss}.", 4, "semantics", [nid])

    # Try: find root → meaning
    for src, _, d in kg.g.in_edges(nid, data=True):
        if d.get("type") == "root_of":
            rn = kg.g.nodes.get(src, {})
            rdisplay = rn.get("display", src)
            for _, mdst, md in kg.g.edges(src, data=True):
                if md.get("type") == "means":
                    meaning = kg.g.nodes.get(mdst, {}).get("display", mdst)
                    return GraphResult(
                        f"{display} carries the root {rdisplay} which means \u201c{meaning}\u201d. "
                        f"This is why {display} relates to \u201c{meaning}\u201d.",
                        4, "semantics", [nid, src, mdst])

    return GraphResult(
        f"I know {display} is in the graph but I don't have its meaning yet. "
        f"Can you ask about it differently?",
        1, "semantics", [nid])


def _graph_route_discovery(kg: KnowledgeGraph, word: str) -> GraphResult:
    """word → prerequisite_for → next words, OR root → used_in → all words."""
    nid = _fuzzy_find(kg, word)
    if not nid:
        return _NO_ANSWER
    node = kg.g.nodes[nid]
    display = node.get("display", nid)

    # Check for prerequisite_for edges
    next_words: list[str] = []
    for _, dst, d in kg.g.edges(nid, data=True):
        if d.get("type") == "prerequisite_for":
            nd = kg.g.nodes.get(dst, {}).get("display", dst)
            next_words.append(nd)

    if next_words:
        return GraphResult(
            f"Now that you know {display}, you are ready to discover: "
            f"{', '.join(next_words[:4])}. Each one shares a root with {display}.",
            4, "discovery", [nid] + next_words)

    # Fallback: via root → used_in
    for src, _, d in kg.g.in_edges(nid, data=True):
        if d.get("type") == "root_of":
            rn = kg.g.nodes.get(src, {})
            rdisplay = rn.get("display", src)
            family: list[str] = []
            for _, dst, d2 in kg.g.edges(src, data=True):
                if d2.get("type") == "used_in" and dst != nid:
                    fd = kg.g.nodes.get(dst, {}).get("display", dst)
                    if fd not in family:
                        family.append(fd)
            if family:
                return GraphResult(
                    f"{display} belongs to the {rdisplay} family. "
                    f"Other words you can discover: {', '.join(family[:6])}.",
                    3, "discovery", [nid, src] + family)

    return _NO_ANSWER


def _graph_route_etymology(kg: KnowledgeGraph, word: str) -> GraphResult:
    """word → root → cognates + translations."""
    nid = _fuzzy_find(kg, word)
    if not nid:
        return _NO_ANSWER
    display = kg.g.nodes[nid].get("display", nid)

    # Find root → cognate_with / translates_to edges
    roots_found: list[str] = []
    for src, _, d in kg.g.in_edges(nid, data=True):
        if d.get("type") == "root_of":
            roots_found.append(src)

    if not roots_found:
        return _NO_ANSWER

    parts: list[str] = []
    for rk in roots_found:
        rdisplay = kg.g.nodes[rk].get("display", rk)
        found_langs: list[tuple[str, str, str]] = []  # (language, word, edge_type)
        for _, dst, d in kg.g.edges(rk, data=True):
            etype = d.get("type", "")
            if etype in ("cognate_with", "translates_to"):
                ln = kg.g.nodes.get(dst, {})
                ld = ln.get("display", dst)
                ltype = ln.get("type", "")
                # Get language from the language-word node description or edge
                lang = etype.replace("_", " ").replace("with", "").strip()
                if "cognate" in etype:
                    lang = "an ancient shared word"
                elif "translate" in etype:
                    lang = "translated"
                found_langs.append((lang, ld, etype))

        if found_langs:
            parts.append(f"The root {rdisplay} appears in {display}. ")
            for lang, lword, _ in found_langs:
                if "cognate" in _:
                    parts.append(f"In Sanskrit it's {lword} — they share the same ancient ancestor. ")
                else:
                    parts.append(f"In Hindi/Bengali: {lword}. ")

    if parts:
        return GraphResult("".join(parts), 4, "etymology", [nid] + roots_found)
    return GraphResult(
        f"{display} comes from the root {kg.g.nodes[roots_found[0]].get('display', roots_found[0])}."
        f" I don't have its etymology details yet.",
        2, "etymology", [nid] + roots_found)


def _graph_route_family(kg: KnowledgeGraph, word: str) -> GraphResult:
    """word → root → used_in → all family words with meanings."""
    nid = _fuzzy_find(kg, word)
    if not nid:
        return _NO_ANSWER
    display = kg.g.nodes[nid].get("display", nid)

    roots_found: list[str] = []
    for src, _, d in kg.g.in_edges(nid, data=True):
        if d.get("type") == "root_of":
            roots_found.append(src)

    if not roots_found:
        return _NO_ANSWER

    parts: list[str] = []
    for rk in roots_found:
        rdisplay = kg.g.nodes[rk].get("display", rk)
        family: list[tuple[str, str]] = []  # (display, gloss)
        for _, dst, d in kg.g.edges(rk, data=True):
            if d.get("type") == "used_in":
                fn = kg.g.nodes.get(dst, {})
                fd = fn.get("display", dst)
                gloss = (fn.get("descriptions", [""]) or [""])[0]
                if fd not in [f[0] for f in family]:
                    family.append((fd, gloss))
        if family:
            parts.append(f"The {rdisplay} word family: ")
            parts.append("; ".join(f"{w} ({g})" if g else w for w, g in family[:8]))
            parts.append(".")

    if parts:
        return GraphResult("".join(parts), 4, "family", [nid] + roots_found)
    return _NO_ANSWER


# ── Main router ─────────────────────────────────────────────────────────────

class GraphRouter:
    """Entry point for the graph-first query engine."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def route(self, query: str, grade: int = 2) -> GraphResult:
        """Attempt to answer entirely from the graph. Returns confidence 0 if
        the graph has nothing — caller should fall through to vector search + LLM."""
        q = query.strip()
        if not q or len(q) < 3:
            return _NO_ANSWER

        intent = classify_educational_intent(q)
        candidates = _find_words_in_query(self.kg, q)

        # Route by intent, trying each candidate until one succeeds
        if candidates:
            if intent == "morphology":
                for c in candidates[:3]:
                    r = _graph_route_morphology(self.kg, c)
                    if r.confidence >= 3:
                        return r
            if intent == "semantics":
                for c in candidates[:3]:
                    r = _graph_route_semantics(self.kg, c)
                    if r.confidence >= 3:
                        return r
            if intent == "discovery":
                for c in candidates[:3]:
                    r = _graph_route_discovery(self.kg, c)
                    if r.confidence >= 3:
                        return r
            if intent == "etymology":
                for c in candidates[:3]:
                    r = _graph_route_etymology(self.kg, c)
                    if r.confidence >= 3:
                        return r
            if intent == "family":
                for c in candidates[:3]:
                    r = _graph_route_family(self.kg, c)
                    if r.confidence >= 3:
                        return r

        # Phonics doesn't need word candidates (works on alias patterns)
        if intent == "phonics":
            return _graph_route_phonics(self.kg, q)

        # Reversal 1: readiness / gap analysis
        if intent == "readiness" and candidates:
            from graph_discovery import gap_analysis
            for c in candidates[:3]:
                ga = gap_analysis(c, mastered_words=[])
                if ga.get("confidence", 0) >= 3:
                    return GraphResult(
                        ga["chain_text"], ga["confidence"], "readiness",
                        [c] + ga.get("prereqs_missing", []),
                        traceback=_build_traceback(
                            target=c, source=f"gap analysis — {ga['prereqs_met']}/{ga['prereqs_total']} prerequisites met"
                        )
                    )

        # Ambiguous query — try morphology as catch-all with best candidate
        if candidates:
            r = _graph_route_morphology(self.kg, candidates[0])
            if r.confidence >= 3:
                return r

        return _NO_ANSWER


# ── Module-level singleton ──────────────────────────────────────────────────

_router: Optional[GraphRouter] = None
_cache: Optional[dict] = None


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        from graph_cache import load_cache
        _cache = load_cache()
    return _cache


def _build_traceback(*, target: str = "", root: str = "", meaning: str = "",
                      phoneme: str = "", graphemes: list | None = None,
                      family: list | None = None,
                      source: str = "graph traversal") -> str:
    """Reversal 3 — render the exact graph path that produced this answer."""
    lines = [f"TARGET: {target}"]
    if root:
        lines.append(f"  \u2191 root_of")
        lines.append(f"  {root} (root" + (f", means \u201c{meaning}\u201d" if meaning else "") + ")")
    if meaning and not root:
        lines.append(f"  \u2191 means")
        lines.append(f"  {meaning} (meaning)")
    if phoneme:
        lines.append(f"  \u2191 sounds_like")
        lines.append(f"  {phoneme} (phoneme" + (f", written as {', '.join(graphemes[:4])}" if graphemes else "") + ")")
    if family:
        lines.append(f"  \u2193 family siblings")
        for f in family[:5]:
            lines.append(f"    {f}")
    lines.append(f"Confidence: deterministic — {source}")
    return "\n".join(lines)


def _cache_lookup(query: str) -> Optional[GraphResult]:
    """O(1) precomputed cache lookup — bypasses graph traversal entirely."""
    # Reversal 1: readiness queries need full gap analysis, not cache shortcut
    if classify_educational_intent(query) == "readiness":
        return None

    cache = _load_cache()
    if not cache:
        return None
    words = cache.get("words", {})

    # Try exact match on the whole query (if it's a single word)
    q = normalize(query)
    for mkey in [q, _singularize(q), q + "s"]:
        if mkey in words:
            w = words[mkey]
            intent = classify_educational_intent(query)
            return _render_from_cache(w, mkey, intent)

    # Try each word in the query
    tokens = re.findall(r"[a-z]{3,}", query.lower())
    for t in tokens:
        n = normalize(t)
        if n in words:
            w = words[n]
            intent = classify_educational_intent(query)
            return _render_from_cache(w, n, intent)
        # Try singular variant
        sn = _singularize(n)
        if sn in words:
            w = words[sn]
            intent = classify_educational_intent(query)
            return _render_from_cache(w, sn, intent)

    return None


def _render_from_cache(w: dict, key: str, intent: str) -> GraphResult:
    """Render a deterministic answer from the precomputed cache entry."""
    display = w.get("display", key)
    morph = w.get("morphology", {})
    phonics = w.get("phonics", {})
    family = w.get("family", [])
    discovery = w.get("discovery", {})
    errors = w.get("errors", {})
    root = morph.get("root", "")
    meaning = morph.get("meaning", "")
    gloss = w.get("gloss", "")

    trace = _build_traceback(
        target=display,
        root=root, meaning=meaning,
        phoneme=phonics.get("phoneme", ""),
        graphemes=phonics.get("graphemes", []),
        family=family, source="precomputed cache (O(1) lookup)",
    )

    if intent == "morphology":
        gloss = w.get("gloss", "")
        parts = [f"{display} comes from the root {root}"]
        if meaning:
            parts.append(f" which means \u201c{meaning}\u201d")
        parts.append(".")
        if gloss:
            parts.append(f" {gloss}.")
        if family:
            siblings = [s for s in family if normalize(s) != normalize(display)]
            if siblings:
                parts.append(f" Words in the same family: {', '.join(siblings[:6])}.")
        return GraphResult("".join(parts), 5, "morphology",
                          [key, morph.get("root_key", "")] + family[:6],
                          traceback=trace)

    if intent == "semantics":
        gloss = w.get("gloss", "")
        if not gloss:
            root = morph.get("root", "")
            meaning = morph.get("meaning", "")
            if root and meaning:
                return GraphResult(
                    f"{display} carries the root {root} which means \u201c{meaning}\u201d.",
                    4, "semantics", [key], traceback=trace)
            return GraphResult(
                f"I know {display} is in the graph but I don't have its meaning yet.",
                1, "semantics", [key], traceback=trace)
        return GraphResult(f"{display}: {gloss}.", 4, "semantics", [key], traceback=trace)

    if intent == "family":
        if family:
            return GraphResult(
                f"The {root} word family"
                f"{' (' + meaning + ')' if meaning else ''}: "
                f"{'; '.join(f for f in family[:8])}.",
                4, "family", [key] + family[:6], traceback=trace)
        return GraphResult(f"{display} belongs to the {root} family.", 3, "family", [key], traceback=trace)

    if intent == "discovery":
        next_words = discovery.get("next", [])
        if next_words:
            return GraphResult(
                f"Now that you know {display}, you are ready to discover: "
                f"{', '.join(next_words[:4])}.",
                4, "discovery", [key] + next_words[:4], traceback=trace)
        if family:
            return GraphResult(
                f"{display} belongs to the {morph.get('root', 'unknown')} family. "
                f"Explore: {', '.join(f for f in family[:6] if normalize(f) != normalize(display))}.",
                3, "discovery", [key], traceback=trace)

    if intent == "etymology":
        if root:
            return GraphResult(
                f"{display} comes from the root {root}"
                f"{' meaning ' + chr(0x201c) + meaning + chr(0x201d) if meaning else ''}.",
                3, "etymology", [key], traceback=trace)
        return GraphResult(f"{display} is in the graph.", 1, "etymology", [key], traceback=trace)

    # Default: morphology-style answer
    parts = []
    if root:
        parts.append(f"{display} comes from {root}")
        if meaning:
            parts.append(f" ({meaning})")
        parts.append(". ")
    if gloss:
        parts.append(f"{gloss}. ")
    if errors.get("watch_for"):
        parts.append(errors["watch_for"])
    if not parts:
        return _NO_ANSWER
    return GraphResult("".join(parts), 4, intent, [key], traceback=trace)


def get_router() -> GraphRouter:
    global _router
    if _router is None:
        from config import GRAPH_PATH
        _router = GraphRouter(KnowledgeGraph.load(GRAPH_PATH))
    return _router


def route_query(query: str, grade: int = 2) -> GraphResult:
    """One-shot graph routing with O(1) cache-first lookup.

    1. Try precomputed cache (O(1) dictionary lookup — ~1μs)
    2. Fall back to graph traversal (O(edges) — sub-ms)
    3. Fall through to vector search + LLM (caller's job)
    """
    # Phase 3: cache-first (O(1) lookup)
    result = _cache_lookup(query)
    if result and result.confidence >= 3:
        return result

    # Phase 2: graph traversal (sub-ms)
    if result is None:
        result = get_router().route(query, grade)
    return result
