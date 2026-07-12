"""Companion voice — rule-based warmth over the twin's memory. No LLM.

A human tutor greets you, remembers what you built yesterday, and narrates
where the journey is. All of that is deterministic templates filled from the
evidence table and the expedition state. Selection is seeded by
(student, day, slot) so the voice feels alive but stable within a session —
the same child gets the same greeting all morning, a fresh one tomorrow.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
import time

from evidence_graph import DB_PATH


def _pick(options: list[str], seed: str) -> str:
    return options[hash(seed) % len(options)]


def _memory(student_id: str) -> dict:
    """What do we remember about this child? Pulled from evidence, not invented."""
    with sqlite3.connect(DB_PATH) as c:
        last = c.execute(
            "SELECT ts FROM evidence WHERE student_id=? ORDER BY ts DESC LIMIT 1",
            (student_id,)).fetchone()
        built = c.execute(
            "SELECT meta, ts FROM evidence WHERE student_id=? AND meta LIKE '%learner_answer%' "
            "ORDER BY ts DESC LIMIT 1", (student_id,)).fetchone()
        n_total = c.execute(
            "SELECT COUNT(DISTINCT node_id) FROM evidence WHERE student_id=?",
            (student_id,)).fetchone()[0]
        streak = c.execute(
            "SELECT outcome FROM evidence WHERE student_id=? ORDER BY ts DESC LIMIT 3",
            (student_id,)).fetchall()
    out = {"last_seen_ts": last[0] if last else None,
           "nodes_met": n_total,
           "last_built": None,
           "hot_streak": len(streak) == 3 and all(r[0] == "correct" for r in streak)}
    if built:
        try:
            out["last_built"] = json.loads(built[0]).get("learner_answer")
        except Exception:
            pass
    return out


def companion_block(student_id: str, expedition=None) -> dict:
    """Greeting + memory line + arc line, ready for the UI's speech bubble."""
    mem = _memory(student_id)
    day = datetime.date.today().isoformat()
    seed = student_id + day

    hour = datetime.datetime.now().hour
    daypart = "morning" if hour < 12 else ("afternoon" if hour < 17 else "evening")

    if mem["last_seen_ts"] is None:
        greeting = _pick([
            f"Good {daypart}! I'm Vaaani. I don't know you yet — every mission you try teaches me how you learn.",
            f"Hello, explorer. Your universe is empty right now. Let's put the first star in it.",
        ], seed + "g")
    else:
        gap_h = (time.time() - mem["last_seen_ts"]) / 3600
        if gap_h < 1:
            greeting = _pick(["Back already — I like it. Let's keep going.",
                              "Still warmed up! Next one's ready."], seed + "g")
        elif gap_h < 30:
            greeting = _pick([f"Good {daypart}! I kept your map exactly where you left it.",
                              f"Welcome back. Your universe has {mem['nodes_met']} stars so far — let's add more."], seed + "g")
        else:
            greeting = _pick([f"It's been a while! Don't worry — I remembered everything, even the tricky ones.",
                              f"Good {daypart}. Some sounds may have gone quiet since last time; I've planned a little revisit."], seed + "g")

    memory_line = ""
    if mem["last_built"]:
        memory_line = _pick([
            f"I still have “{mem['last_built']}” — the word you built. It's in your universe now.",
            f"Last time you invented “{mem['last_built']}”. A real word-builder's move.",
        ], seed + "m")
    elif mem["hot_streak"]:
        memory_line = "Three right in a row last time — your map is growing fast."

    arc_line = ""
    if expedition is not None:
        if expedition.status == "active":
            left = expedition.steps_total - expedition.steps_done
            arc_line = (f"Expedition: the {expedition.display} world — "
                        f"step {expedition.steps_done + 1} of {expedition.steps_total}. "
                        + ("This is the last one — the world lights up after this!" if left == 1
                           else f"{left} discoveries until it lights up."))
        else:
            arc_line = f"🌟 The {expedition.display} world is LIT. Choose your next expedition by playing on."

    return {"greeting": greeting, "memory_line": memory_line, "arc_line": arc_line}
