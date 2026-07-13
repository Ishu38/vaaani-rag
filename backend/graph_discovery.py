#!/usr/bin/env python3
"""Phase 4 — Deterministic Discovery Engine.

Replaces LLM-driven mission selection with graph traversal.  The discovery
engine takes a learner profile (mastered sounds, word families, grade,
weak patterns) and walks the Phase 1 structural linguistics graph to find
the ONE next concept in the learner's Zone of Proximal Development.

Algorithm:
  1. Find words in the graph the learner has mastered (by sound/family match)
  2. Traverse prerequisite_for edges to find reachable next words
  3. Score candidates: prefer words at the learner's grade, connected to
     multiple mastered words, targeting weak patterns
  4. Return the top candidate with a deterministic mission text

Confidence:
  HIGH (5): multiple mastered words point to the same next word
  MEDIUM (3): one mastered word leads to one new word
  LOW (1): no mastered words found — fall back to LLM

This is the "Discovery Graph" + "Learner Graph" from the architecture.
It makes the orchestrator's mission selection deterministic for all
learners whose mastered concepts are in the curriculum graph.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import GRAPH_CACHE_PATH
from graph import normalize


# ── Load graph cache at import time ─────────────────────────────────────────

def _load() -> dict:
    if GRAPH_CACHE_PATH.exists():
        return json.loads(GRAPH_CACHE_PATH.read_text())
    return {}

_cache = _load()


# ── Discovery result ────────────────────────────────────────────────────────

@dataclass
class DiscoveryResult:
    target_word: str
    target_gloss: str = ""
    target_grade: Optional[int] = None
    root: str = ""
    root_meaning: str = ""
    mastered_anchors: list[str] = field(default_factory=list)
    family_siblings: list[str] = field(default_factory=list)
    confidence: int = 0  # 0-5 (5 = best match)
    mission_text: str = ""


# ── Sound → word mapping ────────────────────────────────────────────────────

def _words_for_sound(sound: str) -> list[str]:
    """Find words in the cache whose first phoneme matches the given sound."""
    words = _cache.get("words", {})
    out = []
    for nid, w in words.items():
        phonics = w.get("phonics", {})
        phoneme_key = phonics.get("phoneme_key", "")
        if phoneme_key and phoneme_key.endswith(f"-{sound}"):
            out.append(nid)
    return out


def _words_for_family(family_word: str) -> list[str]:
    """Find words in the cache that belong to a given word family."""
    words = _cache.get("words", {})
    fw_n = normalize(family_word)
    if fw_n in words:
        return [w for w in words[fw_n].get("family", []) if normalize(w) != fw_n]
    return []


def _word_looks_like(query: str) -> Optional[str]:
    """Find a cached word that fuzzy-matches the query."""
    words = _cache.get("words", {})
    q = normalize(query)
    if q in words:
        return q
    # Try singular/plural
    if q.endswith("s") and q[:-1] in words:
        return q[:-1]
    if q + "s" in words:
        return q + "s"
    # Substring match
    for nid in words:
        if q in nid or nid in q:
            return nid
    return None


# ── Discovery algorithm ─────────────────────────────────────────────────────

def discover(context: dict) -> DiscoveryResult:
    """Find the ONE best next word for this learner.

    `context` is the discovery_context dict: mastered_sounds,
    unlocked_word_families, grade, weak_patterns, recent_errors, etc.
    """
    words = _cache.get("words", {})
    roots = _cache.get("roots", {})

    mastered = list(context.get("mastered_sounds") or [])
    families = list(context.get("unlocked_word_families") or [])
    weak = list(context.get("weak_patterns") or context.get("current_weak_areas") or [])
    grade = int(context.get("grade", 1))

    # ── Step 1: Find mastered word nodes ───────────────────────────────
    mastered_words: list[str] = []
    mastered_word_set: set[str] = set()

    # From mastered sounds
    for sound in mastered:
        for w in _words_for_sound(sound):
            if w not in mastered_word_set:
                mastered_words.append(w)
                mastered_word_set.add(w)

    # From unlocked word families
    for fw in families:
        w = _word_looks_like(fw)
        if w and w in words and w not in mastered_word_set:
            mastered_words.append(w)
            mastered_word_set.add(w)
        # Also include family siblings as "known" context
        for sib in _words_for_family(fw):
            if sib not in mastered_word_set:
                mastered_words.append(sib)
                mastered_word_set.add(sib)

    if not mastered_words:
        return DiscoveryResult(target_word="", confidence=0,
                              mission_text="")

    # ── Step 2: Find candidate next words via prerequisite_for ────────
    # For each mastered word, collect what comes next
    candidate_scores: dict[str, int] = defaultdict(int)
    word_anchors: dict[str, list[str]] = defaultdict(list)

    for mw in mastered_words[:16]:  # cap at 16: keep it fast
        if mw not in words:
            continue
        wdata = words[mw]
        next_words = wdata.get("discovery", {}).get("next", [])
        for nw in next_words:
            nw_n = normalize(nw)
            candidate_scores[nw_n] += 1
            word_anchors[nw_n].append(wdata.get("display", mw))

    if not candidate_scores:
        # No prerequisite_for edges — try family expansion instead
        for mw in mastered_words[:8]:
            if mw not in words:
                continue
            family = words[mw].get("family", [])
            for fw in family:
                fw_n = normalize(fw)
                if fw_n not in mastered_word_set:
                    candidate_scores[fw_n] += 1
                    word_anchors[fw_n].append(words[mw].get("display", mw))

    if not candidate_scores:
        return DiscoveryResult(target_word="", confidence=0,
                              mission_text="")

    # ── Step 3: Score with ZPD weighting ───────────────────────────────
    scored: list[tuple[int, int, str]] = []
    for nid, score in candidate_scores.items():
        if nid not in words:
            continue
        wdata = words[nid]
        wgrade = wdata.get("grade") or 99
        # Boost: at learner's grade
        grade_boost = 2 if wgrade == grade else (1 if wgrade <= grade else 0)
        # Boost: targets weak pattern
        weak_boost = 0
        morph = wdata.get("morphology", {})
        root = morph.get("root", "").lower()
        meaning = morph.get("meaning", "").lower()
        for wp in weak:
            wp_l = wp.lower().replace("/", "").replace("→", " ").replace("->", " ")
            if root in wp_l or meaning in wp_l or (root and any(r in wp_l for r in root.split())):
                weak_boost = 3
                break
        total = score * 2 + grade_boost + weak_boost
        scored.append((total, score, nid))

    scored.sort(reverse=True)
    _, best_score, best_word = scored[0]
    wdata = words[best_word]
    anchors = word_anchors[best_word][:3]

    # ── Step 4: Build mission text ─────────────────────────────────────
    morph = wdata.get("morphology", {})
    root = morph.get("root", "")
    meaning = morph.get("meaning", "")
    gloss = wdata.get("gloss", "")
    display = wdata.get("display", best_word)
    family = wdata.get("family", [])
    display_morph = wdata.get("display", best_word)

    # Determine mission type and build text
    conf = min(5, scored[0][0] // 2)

    mission_lines: list[str] = []

    if anchors:
        mission_lines.append(f"You have mastered: {', '.join(anchors[:3])}. ")
    mission_lines.append(f"Today's Discovery Mission")

    if meaning:
        mission_lines.append(f"\n\nThe root {root} means \u201c{meaning}\u201d. "
                            f"Can you discover what {display} means? "
                            f"Look at the word carefully — the answer is hiding inside it.")

    # Add family context for clues
    if family:
        siblings = [f for f in family if normalize(f) != normalize(best_word)][:4]
        if siblings:
            mission_lines.append(f"\n\nClue: Words in the same family as {display} are "
                                f"{', '.join(siblings)}. They all share the piece {root}.")

    # Add error guard
    errors = wdata.get("errors", {})
    if errors.get("watch_for"):
        mission_lines.append(f"\n\n{errors['watch_for']}")

    mission_lines.append(f"\n\nTake your time. I'll give a clue only if you need one.")

    mission = "".join(mission_lines)

    return DiscoveryResult(
        target_word=best_word,
        target_gloss=gloss,
        target_grade=wdata.get("grade"),
        root=root,
        root_meaning=meaning,
        mastered_anchors=anchors,
        family_siblings=[f for f in family if normalize(f) != normalize(best_word)][:5],
        confidence=conf,
        mission_text=mission,
    )


# ── Module-level API ────────────────────────────────────────────────────────

def discover_from_context(context: dict, student_id: str = "") -> dict:
    """Return a dict with mission text and discoverable word info.

    Twin-first: if a student_id is provided, the cognitive twin's ZPD
    frontier takes priority over the static cache. Falls back to the
    graph cache when the twin has no data (cold-start)."""
    # ── Twin-first path ─────────────────────────────────────────
    if student_id:
        try:
            from development_engine import WorldModel, frontier
            import cognitive_twin as twin
            world = WorldModel()
            zpd = frontier(student_id, world, limit=5)
            if zpd:
                snap = twin.snapshot(student_id)
                mastered = [n for n, b in snap.items() if b.mastered][:3]
                anchors = [world.display(n) for n in mastered]
                best = zpd[0]
                result = discover(context)
                result.target_word = best.node_id
                result.mastered_anchors = anchors or result.mastered_anchors
                result.confidence = min(5, result.confidence + 1)
                return {
                    "target_word": result.target_word,
                    "target_gloss": result.target_gloss,
                    "root": result.root,
                    "root_meaning": result.root_meaning,
                    "mastered_anchors": result.mastered_anchors,
                    "family_siblings": result.family_siblings,
                    "confidence": result.confidence,
                    "mission_text": result.mission_text,
                }
        except Exception:
            pass

    # ── Fallback: static cache path ──────────────────────────────
    result = discover(context)
    return {
        "target_word": result.target_word,
        "target_gloss": result.target_gloss,
        "root": result.root,
        "root_meaning": result.root_meaning,
        "mastered_anchors": result.mastered_anchors,
        "family_siblings": result.family_siblings,
        "confidence": result.confidence,
        "mission_text": result.mission_text,
    }


if __name__ == "__main__":
    # Quick smoke test with Aarav's profile
    ctx = {
        "grade": 2,
        "mastered_sounds": ["m", "s", "f", "n"],
        "unlocked_word_families": ["man", "moon", "fish"],
        "weak_patterns": ["ph → /f/"],
        "completed_missions": 41,
    }
    r = discover(ctx)
    print(f"Confidence: {r.confidence}")
    print(f"Target: {r.target_word} (root {r.root})")
    print(f"Anchors: {r.mastered_anchors}")
    print(f"Mission:\n{r.mission_text}")

    # Reversal 1: gap analysis
    print("\n=== REVERSAL 1: Gap Analysis ===")
    tests = ["bilingual", "octet", "solstice", "telescope"]
    for w in tests:
        g = gap_analysis(w, mastered_words=["bicycle", "triangle", "biped"])
        print(f"\n  TARGET: {w}")
        print(f"  Ready: {g['ready_now']} — prereqs met: {g['prereqs_met']}")
        print(f"  Missing: {g['prereqs_missing']} — {len(g['missing_words'])} words needed")
        print(f"  Chain: {g['chain_text']}")


def gap_analysis(target_word: str, mastered_words: list[str] | None = None) -> dict:
    """Reversal 1 — Prerequisite dependency map.
    
    For a given target word, returns what the learner needs to know first.
    Compares prerequisites against mastered_words to show the gap.
    
    Returns:
        ready_now: bool — all prerequisites met
        prereqs_met: int — how many prerequisites the learner knows
        prereqs_total: int — total prerequisites for this word
        prereqs_missing: list[str] — what's still unknown
        chain_text: str — human-readable dependency chain
        confidence: int — 5 = full analysis, 3 = partial, 1 = no data
    """
    words = _cache.get("words", {})
    tn = normalize(target_word)
    if tn not in words:
        return {"ready_now": False, "prereqs_met": 0, "prereqs_total": 0,
                "prereqs_missing": [], "chain_text": f"I don't have data for {target_word} yet.",
                "confidence": 1}

    wdata = words[tn]
    prereq_chain = wdata.get("prerequisite_display", [])
    prereq_keys = wdata.get("prerequisites", [])
    prereq_text = wdata.get("prerequisite_chain", "")

    if not prereq_chain:
        # Check if there's a discovery.prev (words that prerequisite into this)
        prev = wdata.get("discovery", {}).get("prev", [])
        if prev:
            prereq_chain = prev
            prereq_text = f"Before {target_word}, you should know: {' → '.join(prev[:4])}."

    mastered_set = {normalize(m) for m in (mastered_words or [])}
    missing = [p for p in prereq_chain if normalize(p) not in mastered_set]
    met = len(prereq_chain) - len(missing)

    if not prereq_chain:
        return {"ready_now": True, "prereqs_met": 0, "prereqs_total": 0,
                "prereqs_missing": [],
                "chain_text": f"{target_word} has no known prerequisites — you can start exploring it now.",
                "confidence": 3}

    if not missing:
        return {"ready_now": True, "prereqs_met": met, "prereqs_total": len(prereq_chain),
                "prereqs_missing": [],
                "chain_text": f"You're ready for {target_word}! "
                             f"You already know: {' → '.join(prereq_chain[:5])}.",
                "confidence": 5}

    return {"ready_now": False, "prereqs_met": met, "prereqs_total": len(prereq_chain),
            "prereqs_missing": missing,
            "missing_words": [wdata.get("display", tn)] if not mastered_set else missing,
            "chain_text": prereq_text or
                         f"Still needed before {target_word}: {' → '.join(missing[:4])}. "
                         f"You know {met}/{len(prereq_chain)} of the prerequisites.",
            "confidence": 4}
