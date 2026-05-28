"""Graph-aware spaced repetition.

Builds on the existing `student_skills` table (one row per (user, graph
node) with mastery + interval_days + due_at) and adds two things on top:

  1. **Item generation** — given a node, build a review card by finding an
     ingested passage that mentions the node's display name and blanking
     it out (cloze). If no passage matches, fall back to a recall prompt
     drawn from the node's neighbours in the corpus graph.

  2. **Graph-aware selection** — among due-or-overdue items, prefer the
     one that's farthest in graph distance from the last few items the
     user reviewed. This is the interleaving moat: vanilla Anki picks by
     due date alone; we use the constellation structure to choose what
     to surface next so the session keeps moving across themes.
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path

from auth.db import connect
from config import GRAPH_PATH, METADATA_PATH

from . import service as learn


# =========================================================================
#  SM-2-lite grading (4 buttons → mastery + interval update)
# =========================================================================

# Map UI grade → (mastery delta, interval multiplier, minimum interval days)
_GRADE_RULES: dict[str, tuple[float, float, float]] = {
    "again": (-1.5, 0.4, 0.5),   # back to soon
    "hard":  (-0.3, 0.9, 1.0),   # nudge sooner
    "good":  (+0.5, 2.2, 1.0),   # default SM-2 ease for an honest "got it"
    "easy":  (+1.0, 3.5, 2.0),   # push out further
}


def grade_node(user_id: int, node_id: str, display: str, grade: str) -> dict:
    """Apply a 4-button grade to one node and return the updated skill row."""
    grade = grade.lower().strip()
    if grade not in _GRADE_RULES:
        raise ValueError(f"unknown grade '{grade}'")
    mastery_delta, interval_mult, min_interval = _GRADE_RULES[grade]
    key = learn.normalize_topic(node_id) or learn.normalize_topic(display)
    if not key:
        raise ValueError("empty topic key")
    now = datetime.utcnow()
    with connect() as c:
        existing = c.execute(
            "SELECT * FROM student_skills WHERE user_id = ? AND topic = ?",
            (user_id, key),
        ).fetchone()
        if existing:
            mastery = float(existing["mastery"])
            interval = float(existing["interval_days"])
            attempts = int(existing["attempts"]) + 1
        else:
            mastery, interval, attempts = 2.0, 1.0, 1

        mastery = max(0.0, min(5.0, mastery + mastery_delta))
        interval = max(min_interval, min(60.0, interval * interval_mult))
        due_at = now + timedelta(days=interval)

        c.execute(
            """INSERT INTO student_skills (user_id, topic, display, subject, mastery,
                                            interval_days, attempts, last_seen_at, due_at)
                   VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(user_id, topic) DO UPDATE SET
                   display = excluded.display,
                   mastery = excluded.mastery,
                   interval_days = excluded.interval_days,
                   attempts = excluded.attempts,
                   last_seen_at = excluded.last_seen_at,
                   due_at = excluded.due_at""",
            (user_id, key, display or key, None, mastery, interval, attempts,
             now.isoformat(), due_at.isoformat()),
        )
        # Mirror the rating into student_attempts so the dashboards already
        # built on top of that table also reflect review sessions.
        rating_mirror = {"again": -1, "hard": 0, "good": 1, "easy": 1}[grade]
        c.execute(
            "INSERT INTO student_attempts (user_id, topic, rating, query) VALUES (?,?,?,?)",
            (user_id, key, rating_mirror, f"[review: {grade}]"),
        )

    return {
        "topic": key,
        "display": display or key,
        "mastery": mastery,
        "interval_days": interval,
        "attempts": attempts,
        "due_at": due_at.isoformat(),
        "grade": grade,
    }


# =========================================================================
#  Graph + corpus helpers (cached at module load)
# =========================================================================

_GRAPH_CACHE: dict | None = None
_ADJ_CACHE: dict[str, set[str]] | None = None
_NODE_BY_ID: dict[str, dict] | None = None


def _graph() -> tuple[dict[str, dict], dict[str, set[str]]]:
    global _GRAPH_CACHE, _ADJ_CACHE, _NODE_BY_ID
    if _NODE_BY_ID is not None and _ADJ_CACHE is not None:
        return _NODE_BY_ID, _ADJ_CACHE
    if not Path(GRAPH_PATH).exists():
        _NODE_BY_ID, _ADJ_CACHE = {}, defaultdict(set)
        return _NODE_BY_ID, _ADJ_CACHE
    raw = json.loads(Path(GRAPH_PATH).read_text())
    nodes = {str(n["id"]): n for n in raw.get("nodes", []) if n.get("id")}
    adj: dict[str, set[str]] = defaultdict(set)
    for e in raw.get("links", []) or raw.get("edges", []):
        s, t = str(e.get("source", "")), str(e.get("target", ""))
        if s in nodes and t in nodes and s != t:
            adj[s].add(t)
            adj[t].add(s)
    _GRAPH_CACHE = raw
    _NODE_BY_ID = nodes
    _ADJ_CACHE = adj
    return nodes, adj


def invalidate_graph_cache() -> None:
    """Call after re-ingest so the new graph shape is picked up."""
    global _GRAPH_CACHE, _ADJ_CACHE, _NODE_BY_ID
    _GRAPH_CACHE = None
    _ADJ_CACHE = None
    _NODE_BY_ID = None


def _bfs_distance(start: str, target: str, adj: dict[str, set[str]], cap: int = 6) -> int:
    """Shortest-path hop count between two graph nodes; cap+1 if unreachable
    within `cap` hops. Capping keeps the selection loop cheap."""
    if start == target:
        return 0
    seen = {start}
    frontier: deque[tuple[str, int]] = deque([(start, 0)])
    while frontier:
        cur, d = frontier.popleft()
        if d >= cap:
            continue
        for nb in adj.get(cur, ()):
            if nb in seen:
                continue
            if nb == target:
                return d + 1
            seen.add(nb)
            frontier.append((nb, d + 1))
    return cap + 1


# =========================================================================
#  Selection: graph-aware interleaving
# =========================================================================

def _recent_node_ids(user_id: int, limit: int = 3) -> list[str]:
    with connect() as c:
        rows = c.execute(
            "SELECT topic FROM student_attempts WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [r["topic"] for r in rows]


def _due_candidates(user_id: int, limit: int = 30) -> list[dict]:
    """Pull due-or-overdue items for this user, ordered by due_at."""
    now_iso = datetime.utcnow().isoformat()
    with connect() as c:
        rows = c.execute(
            """SELECT topic, display, mastery, interval_days, due_at
                 FROM student_skills
                WHERE user_id = ? AND due_at <= ?
                ORDER BY due_at ASC
                LIMIT ?""",
            (user_id, now_iso, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def _seed_candidates(user_id: int, limit: int = 30) -> list[dict]:
    """Top-degree graph nodes the user hasn't touched yet. Used when the
    user has no tracked skills (first visit to Review)."""
    now_iso = datetime.utcnow().isoformat()
    with connect() as c:
        tracked = {r["topic"] for r in c.execute(
            "SELECT topic FROM student_skills WHERE user_id = ?", (user_id,)
        ).fetchall()}
    nodes, adj = _graph()
    by_degree = sorted(nodes.items(), key=lambda kv: -len(adj.get(kv[0], ())))
    seeded: list[dict] = []
    for nid, n in by_degree:
        if nid in tracked:
            continue
        seeded.append({
            "topic": nid,
            "display": n.get("display", nid),
            "mastery": 2.0,
            "interval_days": 1.0,
            "due_at": now_iso,
        })
        if len(seeded) >= limit:
            break
    return seeded


def select_next(user_id: int) -> dict | None:
    """Pick the next review item using graph-distance interleaving.

    Behaviour:
      * If the user has due-or-overdue items → choose among those, ranked
        by graph distance to recent items (interleaving).
      * Otherwise, if the user has never been tracked → seed with
        top-degree untracked nodes and pick the first.
      * Otherwise (tracked, nothing due) → return None so the UI can
        show "queue empty, come back later".
    """
    candidates = _due_candidates(user_id, limit=40)
    seeded_mode = False
    if not candidates:
        with connect() as c:
            tracked_count = c.execute(
                "SELECT COUNT(*) AS n FROM student_skills WHERE user_id = ?", (user_id,)
            ).fetchone()["n"]
        if tracked_count > 0:
            return None
        candidates = _seed_candidates(user_id, limit=20)
        seeded_mode = True
        if not candidates:
            return None

    recent = _recent_node_ids(user_id, limit=3)
    if not recent:
        return candidates[0]
    _, adj = _graph()

    def score(cand: dict) -> tuple[int, str]:
        nid = cand["topic"]
        min_d = min((_bfs_distance(nid, r, adj) for r in recent), default=99)
        # Larger graph distance from recent items first; then oldest due first.
        return (-min_d, cand["due_at"])

    candidates.sort(key=score)
    pick = candidates[0]
    pick["_seeded"] = seeded_mode
    return pick


# =========================================================================
#  Item generation: cloze from corpus, fallback to recall
# =========================================================================

_CHUNKS_CACHE: list[dict] | None = None


def _chunks() -> list[dict]:
    global _CHUNKS_CACHE
    if _CHUNKS_CACHE is not None:
        return _CHUNKS_CACHE
    if not Path(METADATA_PATH).exists():
        _CHUNKS_CACHE = []
        return _CHUNKS_CACHE
    meta = json.loads(Path(METADATA_PATH).read_text())
    _CHUNKS_CACHE = list(meta.get("chunks", []))
    return _CHUNKS_CACHE


def invalidate_chunks_cache() -> None:
    global _CHUNKS_CACHE
    _CHUNKS_CACHE = None


def _find_cloze_passage(display: str, *, source_filter: set[str] | None = None) -> tuple[str, str, str] | None:
    """Return (raw_passage, cloze_passage, doc_name) or None if no match.
    Picks the chunk where the term has the most balanced context (a
    sentence of ~12-40 words around the first mention).

    ``source_filter`` (optional set of filenames): when supplied, only
    chunks whose ``source`` is in the set are eligible. Used by the
    Review modal's per-source scope chip and by /anki/export?source=…
    so the deck only contains cards from the chosen documents.
    """
    if not display:
        return None
    pattern = re.compile(rf"\b{re.escape(display)}\b", re.IGNORECASE)
    best: tuple[int, str, str, str] | None = None  # (score, raw, cloze, source)
    for ch in _chunks():
        if source_filter and ch.get("source") not in source_filter:
            continue
        text = ch.get("text", "")
        if not text or not pattern.search(text):
            continue
        # Find the sentence containing the term.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sent in sentences:
            if not pattern.search(sent):
                continue
            sent = sent.strip()
            words = sent.split()
            if not (8 <= len(words) <= 60):
                continue
            cloze = pattern.sub("_____", sent)
            # Score: prefer sentences that mention the term exactly once,
            # are medium length, and don't start with the term itself.
            mention_count = len(pattern.findall(sent))
            length_score = -abs(20 - len(words))  # closer to 20 wpc = better
            uniq_bonus = 5 if mention_count == 1 else 0
            start_penalty = -3 if pattern.match(sent) else 0
            score = length_score + uniq_bonus + start_penalty
            if best is None or score > best[0]:
                best = (score, sent, cloze, ch.get("source", ""))
    if best is None:
        return None
    _, raw, cloze, src = best
    return raw, cloze, src


def _build_recall_prompt(node_id: str, display: str) -> dict:
    """Fallback prompt when no cloze sentence is available: ask the
    student to explain how this node relates to one of its neighbours."""
    _, adj = _graph()
    neighbours = list(adj.get(node_id, ()))
    if not neighbours:
        return {
            "node_id": node_id,
            "display": display,
            "mode": "free",
            "prompt": f"Explain {display} in your own words. Mention at least two ideas it depends on.",
            "answer": "",
            "source": "",
        }
    random.shuffle(neighbours)
    nodes, _ = _graph()
    pick = neighbours[0]
    pick_display = nodes.get(pick, {}).get("display", pick)
    return {
        "node_id": node_id,
        "display": display,
        "mode": "recall",
        "prompt": f"In one or two sentences, explain the relationship between **{display}** and **{pick_display}**.",
        "answer": "",
        "source": "",
    }


def build_review_item(node_id: str, display: str, *, source_filter: set[str] | None = None) -> dict:
    """Top-level: return a review card payload ready for the UI."""
    nodes, adj = _graph()
    n = nodes.get(node_id, {})
    descriptions = n.get("descriptions") or []
    if not display:
        display = n.get("display", node_id)
    cloze = _find_cloze_passage(display, source_filter=source_filter)
    if cloze:
        raw, cloze_text, src = cloze
        return {
            "node_id": node_id,
            "display": display,
            "mode": "cloze",
            "prompt": cloze_text,
            "answer": raw,
            "source": src,
            "type": n.get("type", ""),
            "description": (descriptions[0] if descriptions else "")[:200],
        }
    item = _build_recall_prompt(node_id, display)
    item["type"] = n.get("type", "")
    item["description"] = (descriptions[0] if descriptions else "")[:200]
    return item


# =========================================================================
#  Public entry points
# =========================================================================

def next_review(user_id: int, *, source_filter: set[str] | None = None) -> dict | None:
    """Pick the next due item (graph-aware) and build its review card.

    ``source_filter`` (optional): when supplied, the picker scans candidates
    in graph-aware priority order and returns the first one that has a
    cloze passage available in the filtered sources. If none do, returns
    None — UI surfaces "no review material in this source" gracefully.
    """
    # When a source filter is in play we may need to skip the top-priority
    # candidate (if it has no cloze in the chosen source) and walk down.
    if source_filter:
        candidates = _due_candidates(user_id, limit=40) or _seed_candidates(user_id, limit=40)
        for cand in candidates:
            item = build_review_item(cand["topic"], cand["display"], source_filter=source_filter)
            if item.get("mode") == "cloze" and item.get("source") in source_filter:
                item["mastery"] = round(float(cand.get("mastery", 2.0)), 2)
                item["interval_days"] = round(float(cand.get("interval_days", 1.0)), 2)
                item["due_at"] = cand.get("due_at", "")
                return item
        return None
    pick = select_next(user_id)
    if pick is None:
        return None
    item = build_review_item(pick["topic"], pick["display"])
    item["mastery"] = round(float(pick.get("mastery", 2.0)), 2)
    item["interval_days"] = round(float(pick.get("interval_days", 1.0)), 2)
    item["due_at"] = pick.get("due_at", "")
    return item


def session_stats(user_id: int) -> dict:
    """Quick header stats for the review modal."""
    now_iso = datetime.utcnow().isoformat()
    with connect() as c:
        due = c.execute(
            "SELECT COUNT(*) AS n FROM student_skills WHERE user_id=? AND due_at<=?",
            (user_id, now_iso),
        ).fetchone()
        tracked = c.execute(
            "SELECT COUNT(*) AS n FROM student_skills WHERE user_id=?",
            (user_id,),
        ).fetchone()
        today_reviews = c.execute(
            """SELECT COUNT(*) AS n FROM student_attempts
                 WHERE user_id=? AND query LIKE '[review:%' AND at >= date('now')""",
            (user_id,),
        ).fetchone()
    return {
        "due_now": int(due["n"]) if due else 0,
        "tracked": int(tracked["n"]) if tracked else 0,
        "reviewed_today": int(today_reviews["n"]) if today_reviews else 0,
    }
