"""Vaaani Discovery Orchestrator — the adaptive learning engine.

The orchestrator decides what the learner should discover next. It reads the
learner's current state (grade, mastered sounds, unlocked word families, weak
areas, recent discoveries) and generates ONE exploration mission perfectly
matched to their Zone of Proximal Development.

It is not a chatbot, not a tutor, not a search engine. It engineers discovery.
Every completed mission unlocks the next automatically — the learner never has
to decide what to ask next.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import COMMUNITIES_PATH, DATA_DIR, GRAPH_PATH


# ── The Orchestrator System Prompt ────────────────────────────────────
# Injected as the sole system prompt when the learner is in discovery mode.
# It replaces the standard tutor persona. The prompt is abridged from the
# full design document to fit within a 4B model's reliable attention span
# while preserving the core behavioral rules.

DISCOVERY_ORCHESTRATOR = """# IDENTITY

You are Vaaani Discovery Orchestrator — not a chatbot, not a tutor, not a search engine. You are the adaptive learning engine that decides what the learner should discover next. Your purpose: transform every learner into an independent language explorer. You never wait for curiosity — you create it.

# PRIMARY RESPONSIBILITY

The learner should NEVER think "What should I ask?" They immediately receive an exploration challenge perfectly matched to what they know. Every session begins with discovery — never conversation.

# LEARNING PHILOSOPHY

Knowledge is discovered, not delivered: Observe → Compare → Predict → Experiment → Discover → Apply → Master.

# FIRST RESPONSE OF EVERY SESSION

Never greet with "Hello", "How can I help?", or "What would you like to learn?"
1. Read the LEARNER STATE.
2. Determine what they recently mastered.
3. Determine what naturally comes next.
4. Generate ONE exploration mission.

# MISSION RULES

1. Build directly on previously mastered knowledge.
2. Introduce only ONE new idea.
3. Feel like solving a mystery.
4. Work inside the Zone of Proximal Development: ~80% familiar, ~20% new. Never overwhelm. Never bore.

# DISCOVERY LOOP

1. Present the challenge. 2. Wait — do NOT explain. 3. If the learner struggles, offer ONE clue (not the answer). 4. Allow another attempt. 5. Offer another clue. 6. Only after genuine effort, explain why. 7. Immediately create another mission using the newly acquired knowledge. Learning never stops after an answer.

# WHEN THE LEARNER MAKES AN ERROR

Never immediately correct. Ask: "Read it aloud. What do you hear? Does anything surprise you? Can you compare it with another word?" Allow discovery before correction.

# WHEN A LEARNER ASKS A DIRECT QUESTION

Do not answer directly. Turn it into a mission. "Look carefully at these words — what pattern do you notice?" Only reveal the explanation after the learner has explored.

# PRAISE

Praise observations, not intelligence: "Interesting observation." "That's an important clue." "You're noticing patterns." "Nice comparison." Never say "You're smart" or "You're brilliant."

# GRAPH-RAG

Convert retrieved knowledge into comparisons, patterns, or mysteries. Never dump retrieved information.

# CURIOSITY ENGINE

Every response should create exactly one unanswered question. Every solved mystery should unlock another. Learning should feel like exploring an infinite map.

# END OF SESSION

Never simply end. Unlock tomorrow: "Before our next mission, see if you can discover another word that breaks today's pattern."

# SUCCESS METRIC

Success is NOT measured by correct answers. Success is measured by increasing curiosity.

# DEEP-LINK TARGETS

When a mission needs a tool, include: [[LINK:/path|Label]]
Targets: /sound-lab?s=<sound> (Hear/feel a sound), /feel (haptic buzz), /graph-view (knowledge graph), /roots?root=<root> (word family), /explore (camera).
At most ONE link per mission. Only when the mission genuinely needs a tool.

# FINAL PRINCIPLE

Vaaani does not answer questions. Vaaani engineers discovery. Every completed mission unlocks the next automatically.

# FORMAT

2–4 sentences. ONE challenge. End with an invitation to explore, not an answer. When the learner has not yet attempted, present the mission and WAIT."""


# ── Learner State Builder ─────────────────────────────────────────────

def build_discovery_state(
    user: dict | None,
    discovery_context: dict | None = None,
) -> str:
    """Build the LEARNER STATE block injected into the orchestrator prompt.

    Combines three sources (priority order):
    1. `discovery_context` — what the calling page passes (grade, sound, L1,
       completed sounds, the page they came from) PLUS the full learner
       profile schema (mastered_sounds, current_unit, completed_missions,
       recent_errors, weak_patterns, unlocked_word_families, age, previously
       asked questions, current session history).
    2. Cognitive fingerprint — strengths, weaknesses, bias, resilience
    3. Knowledge graph — available word families the learner could explore
    """
    ctx = discovery_context or {}
    lines: list[str] = []

    # ── Header ──
    lines.append(
        "LEARNER STATE — the application has provided this summary. "
        "Use it to generate exactly ONE discovery mission. "
        "Do NOT mention this state block to the learner."
    )

    # ── Name ──
    name = ctx.get("name", "")
    if not name and user:
        name = user.get("name") or user.get("display_name") or ""
    if name:
        lines.append(f"Name: {name}.")

    # ── Age (optional) ──
    age = ctx.get("age")
    if age is not None:
        try:
            lines.append(f"Age: {int(age)}.")
        except (TypeError, ValueError):
            pass

    # ── Origin ──
    source = ctx.get("source", "")
    if source:
        source_label = {"ipa": "the IPA Chart (phonetics)", "explore": "the Explore camera", "sound-lab": "the Sound Lab", "graph-view": "the Knowledge Graph"}.get(source, source)
        lines.append(f"Arrived from: {source_label}.")

    # ── Grade / current learning stage ──
    grade = ctx.get("grade")
    if grade is not None:
        try:
            grade = int(grade)
            lines.append(f"Grade: {grade}.")
        except (TypeError, ValueError):
            pass
    current_unit = ctx.get("current_unit") or ctx.get("current_stage") or ""
    if current_unit:
        lines.append(f"Current learning stage: {current_unit}.")

    # ── Native language ──
    l1 = ctx.get("l1") or ctx.get("native_language") or ""
    if l1:
        lines.append(f"Native language: {l1}.")

    # ── What they were just studying ──
    sound = ctx.get("sound", "")
    if sound:
        lines.append(f"Current sound being studied: {sound}.")

    current_topic = ctx.get("topic", "")
    if current_topic:
        lines.append(f"Current topic: {current_topic}.")

    # ── Completed / mastered sounds ──
    completed = ctx.get("completed", [])
    if isinstance(completed, str):
        completed = [s.strip() for s in completed.split(",") if s.strip()]
    mastered = ctx.get("mastered_sounds", [])
    if isinstance(mastered, str):
        mastered = [s.strip() for s in mastered.split(",") if s.strip()]
    if mastered:
        lines.append(f"Mastered sounds: {', '.join(mastered[:12])}.")
    elif completed:
        lines.append(f"Recently completed sounds/patterns: {', '.join(completed[:8])}.")

    # ── Unlocked spelling patterns / word families / roots ──
    patterns = ctx.get("unlocked_spelling_patterns", [])
    if isinstance(patterns, str):
        patterns = [s.strip() for s in patterns.split(",") if s.strip()]
    if patterns:
        lines.append(f"Unlocked spelling patterns: {', '.join(patterns[:8])}.")
    families = ctx.get("unlocked_word_families", [])
    if isinstance(families, str):
        families = [s.strip() for s in families.split(",") if s.strip()]
    if families:
        lines.append(f"Unlocked word families: {', '.join(families[:8])}.")
    roots = ctx.get("unlocked_roots", [])
    if isinstance(roots, str):
        roots = [s.strip() for s in roots.split(",") if s.strip()]
    if roots:
        lines.append(f"Unlocked roots: {', '.join(roots[:8])}.")

    # ── Mission count ──
    missions = ctx.get("completed_missions")
    if missions is not None:
        try:
            lines.append(f"Completed missions: {int(missions)}.")
        except (TypeError, ValueError):
            pass

    # ── Weak areas / recently confused concepts ──
    weak = ctx.get("weak_patterns") or ctx.get("current_weak_areas") or []
    if isinstance(weak, str):
        weak = [s.strip() for s in weak.split(",") if s.strip()]
    if weak:
        lines.append(f"Current weak areas: {', '.join(weak[:6])}.")
    confused = ctx.get("recent_errors") or ctx.get("recently_confused_concepts") or []
    if isinstance(confused, str):
        confused = [s.strip() for s in confused.split(",") if s.strip()]
    if confused:
        lines.append(f"Recently confused concepts: {', '.join(confused[:6])}.")

    # ── Previously asked questions (so we don't repeat) ──
    asked = ctx.get("previously_asked_questions", [])
    if isinstance(asked, str):
        asked = [s.strip() for s in asked.split("|") if s.strip()]
    if asked:
        lines.append(f"Previously asked (do not repeat these): {'; '.join(asked[-5:])}.")

    # ── Current session history (the turns so far this session) ──
    session = ctx.get("current_session_history", [])
    if isinstance(session, list) and session:
        last = session[-3:]
        lines.append("Current session (most recent turns):")
        for t in last:
            role = t.get("role", "?") if isinstance(t, dict) else "?"
            text = (t.get("text") or t.get("content") or "") if isinstance(t, dict) else str(t)
            lines.append(f"  {role}: {text[:160]}")

    # ── Cognitive fingerprint ──
    if user:
        try:
            from cognitive.fingerprint import build_fingerprint
            fp = build_fingerprint(user["id"])
            s = fp.get("summary", {}) or {}
            if s.get("total_analyzed"):
                lines.append(
                    f"Past sessions: {s['total_analyzed']} questions answered, "
                    f"~{s.get('accuracy', 0)}% correct."
                )
            if fp.get("strengths"):
                lines.append(
                    f"Strong in: {', '.join(fp['strengths'][:3])} "
                    "(acknowledge, do not over-drill)."
                )
            if fp.get("weaknesses"):
                lines.append(
                    f"Weak areas: {', '.join(fp['weaknesses'][:3])} "
                    "(gently steer practice here)."
                )
            pw = s.get("primary_weakness_label")
            if pw and pw not in ("None", "No data yet", "Unknown"):
                lines.append(f"Most common mistake type: {pw}.")
        except Exception:
            pass

    # ── Available discovery paths from the knowledge graph ──
    touched_sounds = list(mastered) if mastered else (completed if completed else [])
    graph_paths = _available_discovery_paths(touched_sounds, sound, grade=grade)
    if graph_paths:
        lines.append(f"AVAILABLE DISCOVERY PATHS (from the knowledge graph):")
        for path in graph_paths:
            lines.append(f"  - {path}")

    # ── Available Graph-RAG knowledge (generic blob the app may pass) ──
    rag_knowledge = ctx.get("available_graph_rag_knowledge")
    if rag_knowledge:
        if isinstance(rag_knowledge, list):
            lines.append("AVAILABLE GRAPH-RAG KNOWLEDGE:")
            for k in rag_knowledge[:5]:
                lines.append(f"  - {str(k)[:200]}")
        elif isinstance(rag_knowledge, str):
            lines.append(f"AVAILABLE GRAPH-RAG KNOWLEDGE: {rag_knowledge[:400]}")

    return "\n".join(lines)


def _available_discovery_paths(
    completed_sounds: list[str],
    current_sound: str,
    grade: int | None = None,
) -> list[str]:
    """Scan the knowledge graph for word-family communities the learner
    hasn't yet explored, returning short descriptions usable as mission seeds."""
    paths: list[str] = []
    try:
        if COMMUNITIES_PATH.exists():
            with open(COMMUNITIES_PATH) as f:
                communities = json.load(f)

            # Build a set of all entity keys the learner has touched
            touched = {s.lower().strip() for s in completed_sounds if s.strip()}
            if current_sound:
                touched.add(current_sound.lower().strip())

            for c in communities:
                title = c.get("title", "")
                nodes = c.get("nodes", [])
                if not title or not nodes:
                    continue
                # Check if any node overlaps with learner's touched concepts
                overlap = any(
                    n.lower() in touched or any(t in n.lower() for t in touched)
                    for n in nodes
                )
                if overlap:
                    # The learner knows something in this community —
                    # suggest the next exploration
                    new_nodes = [n for n in nodes if n.lower() not in touched]
                    if new_nodes:
                        paths.append(
                            f"Word family: {title} — "
                            f"you know {len(nodes) - len(new_nodes)}/{len(nodes)} words; "
                            f"still to discover: {', '.join(new_nodes[:4])}"
                        )

            # If no direct overlap, suggest top communities as new territory
            if not paths and communities:
                top = communities[:3]
                for c in top:
                    title = c.get("title", "")
                    nodes = c.get("nodes", [])[:3]
                    if title and nodes:
                        paths.append(
                            f"New territory — word family: {title} "
                            f"(e.g., {', '.join(nodes)})"
                        )

    except Exception:
        pass

    # Phase 5 — Curriculum-aware discovery paths
    if grade is not None:
        try:
            from graph_curriculum import (
                words_for_grade, roots_for_grade,
            )
            curr = "ncert"
            required_words = words_for_grade(curr, int(grade))
            touched_words = {s.lower().strip() for s in completed_sounds if s.strip()}
            remaining_words = [w for w in required_words
                             if w.lower().strip() not in touched_words][:8]
            if remaining_words:
                paths.append(
                    f"CURRICULUM (NCERT Grade {grade}): "
                    f"still to master — {', '.join(remaining_words[:6])}"
                )
        except Exception:
            pass

    return paths[:8]

