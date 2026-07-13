"""Mastery + spaced-repetition service.

Uses the same SQLite database as the auth module (`data/users.db`) and the
`student_skills` / `student_attempts` tables created in `auth/db.py`.

Mastery model (SM-2-lite):
  rating -1 (confused)  →  mastery -= 1, interval = max(1, interval / 2)
  rating  0 (sort of)   →  mastery unchanged, interval stays
  rating +1 (got it)    →  mastery += 1, interval *= 2 (capped at 60 days)
Mastery is clamped to [0, 5]; due_at = last_seen + interval days.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from auth.db import connect


# ---------------- normalisation ----------------

def normalize_topic(name: str) -> str:
    """Canonical key for a topic — same convention as graph.normalize()."""
    return " ".join((name or "").split()).lower()


# Lightweight subject classifier mirrors the Socratic prompt pedagogies
# (linguistics-only scope: sounds / words / sentences / meaning / lit / writing).
_SUBJECT_RULES = [
    ("etymology",  r"\b(etymolog\w*|loanword|borrow\w*|cognate|word origin|latin|greek|sanskrit|proto[- ]?indo)\b"),
    ("phonetics",  r"\b(vowel|consonant|voiced|voiceless|articulat\w*|aspirat\w*|pronunciat\w*|pronounce|accent|ipa)\b"),
    ("phonology",  r"\b(phoneme\w*|syllable\w*|stress|intonation|minimal pairs?|phonotactic\w*|assimilat\w*|rhym\w*)\b"),
    ("morphology", r"\b(morpheme\w*|prefix\w*|suffix\w*|affix\w*|inflect\w*|deriv\w*|plural\w*|compound\w*|word[- ]formation)\b"),
    ("semantics",  r"\b(meaning|synonym\w*|antonym\w*|homonym\w*|polysem\w*|idiom\w*|ambigu\w*|connotation|denotation|presuppos\w*)\b"),
    ("syntax",     r"\b(sentence\w*|clauses?|phrases?|grammar|grammatical|word order|passive|tense|noun|verb|adjective|adverb|pronoun)\b"),
    ("english",    r"\b(stanza|metaphor|simile|alliteration|protagonist|antagonist|theme|imagery|sonnet|allegory|narrator)\b"),
    ("writing",    r"\b(thesis|claim|warrant|evidence|rhetoric|ethos|pathos|logos|counter[- ]?argument|persuad)\b"),
]


def classify_subject(query: str, topic_display: str = "") -> str | None:
    """Best-effort subject tag for a topic, based on the surrounding query."""
    text = f"{query or ''} {topic_display or ''}".lower()
    for subject, pattern in _SUBJECT_RULES:
        if re.search(pattern, text):
            return subject
    return None


# ---------------- mastery updates ----------------

def _now() -> datetime:
    """UTC now with no microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _clamp(v: float, lo: float, hi: float) -> float:
    """Clamp v into [lo, hi]."""
    return max(lo, min(hi, v))


def upsert_skill(user_id: int, topic: str, display: str, subject: str | None) -> None:
    """Touch a skill row when the student engages with the topic (no rating yet)."""
    key = normalize_topic(topic) or normalize_topic(display)
    if not key:
        return
    with connect() as c:
        c.execute(
            """INSERT INTO student_skills (user_id, topic, display, subject, last_seen_at)
                   VALUES (?,?,?,?, ?)
               ON CONFLICT(user_id, topic) DO UPDATE SET
                   display = excluded.display,
                   subject = COALESCE(excluded.subject, student_skills.subject),
                   last_seen_at = excluded.last_seen_at""",
            (user_id, key, display or key, subject, _now().isoformat()),
        )


def record_attempt(
    user_id: int,
    topic: str,
    display: str,
    rating: int,
    query: str | None = None,
    subject: str | None = None,
) -> dict:
    """Apply a single student rating to one topic and return the updated skill row.

    Rating must be -1, 0, or +1.
    """
    rating = max(-1, min(1, int(rating)))
    key = normalize_topic(topic) or normalize_topic(display)
    if not key:
        return {}
    now = _now()
    with connect() as c:
        existing = c.execute(
            "SELECT * FROM student_skills WHERE user_id = ? AND topic = ?", (user_id, key)
        ).fetchone()
        if existing:
            mastery = float(existing["mastery"])
            interval = float(existing["interval_days"])
            attempts = int(existing["attempts"]) + 1
        else:
            mastery, interval, attempts = 2.0, 1.0, 1

        if rating > 0:
            mastery = _clamp(mastery + 1.0, 0.0, 5.0)
            interval = min(60.0, max(1.0, interval * 2.0))
        elif rating < 0:
            mastery = _clamp(mastery - 1.0, 0.0, 5.0)
            interval = max(1.0, interval / 2.0)
        # rating == 0 leaves both unchanged

        due_at = now + timedelta(days=interval)

        c.execute(
            """INSERT INTO student_skills (user_id, topic, display, subject, mastery,
                                            interval_days, attempts, last_seen_at, due_at)
                   VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(user_id, topic) DO UPDATE SET
                   display = excluded.display,
                   subject = COALESCE(excluded.subject, student_skills.subject),
                   mastery = excluded.mastery,
                   interval_days = excluded.interval_days,
                   attempts = excluded.attempts,
                   last_seen_at = excluded.last_seen_at,
                   due_at = excluded.due_at""",
            (user_id, key, display or key, subject, mastery, interval, attempts,
             now.isoformat(), due_at.isoformat()),
        )
        c.execute(
            "INSERT INTO student_attempts (user_id, topic, rating, query) VALUES (?,?,?,?)",
            (user_id, key, rating, (query or "")[:300]),
        )

    return {
        "topic": key,
        "display": display or key,
        "mastery": mastery,
        "interval_days": interval,
        "attempts": attempts,
        "due_at": due_at.isoformat(),
    }


def record_attempts_bulk(
    user_id: int,
    topics: list[tuple[str, str]],
    rating: int,
    query: str | None = None,
) -> list[dict]:
    """Apply the same rating across many topics (used when a Socratic turn touched several entities)."""
    subject = classify_subject(query or "", " ".join(d for _, d in topics))
    out = []
    for topic, display in topics:
        row = record_attempt(user_id, topic, display, rating, query, subject)
        if row:
            out.append(row)
    return out


# ---------------- queries ----------------

def list_skills(user_id: int, limit: int = 200) -> list[dict]:
    """All tracked skills for the dashboard, freshest first."""
    with connect() as c:
        rows = c.execute(
            "SELECT topic, display, subject, mastery, interval_days, attempts, "
            "       last_seen_at, due_at FROM student_skills "
            "WHERE user_id = ? ORDER BY last_seen_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def due_for_review(user_id: int, limit: int = 12) -> list[dict]:
    """Items whose `due_at` is in the past, weakest first."""
    with connect() as c:
        rows = c.execute(
            "SELECT topic, display, subject, mastery, due_at FROM student_skills "
            "WHERE user_id = ? AND due_at <= datetime('now') "
            "ORDER BY mastery ASC, due_at ASC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def weak_spots(user_id: int, candidate_topics: list[str], threshold: float = 2.0, limit: int = 3) -> list[dict]:
    """Filter `candidate_topics` (canonical keys) to those the student has rated weak.

    Used by the Socratic prompt builder to bias questioning toward known weak topics.
    """
    if not candidate_topics:
        return []
    keys = list({normalize_topic(t) for t in candidate_topics if t})
    if not keys:
        return []
    placeholders = ",".join("?" * len(keys))
    with connect() as c:
        rows = c.execute(
            f"SELECT topic, display, mastery FROM student_skills "
            f"WHERE user_id = ? AND mastery <= ? AND topic IN ({placeholders}) "
            f"ORDER BY mastery ASC LIMIT ?",
            (user_id, threshold, *keys, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def stats(user_id: int) -> dict:
    """Aggregate stats for the dashboard hero strip."""
    with connect() as c:
        total = c.execute("SELECT COUNT(*) AS n FROM student_skills WHERE user_id = ?", (user_id,)).fetchone()["n"]
        avg = c.execute(
            "SELECT AVG(mastery) AS m FROM student_skills WHERE user_id = ?", (user_id,)
        ).fetchone()["m"]
        due = c.execute(
            "SELECT COUNT(*) AS n FROM student_skills WHERE user_id = ? AND due_at <= datetime('now')",
            (user_id,),
        ).fetchone()["n"]
        strong = c.execute(
            "SELECT COUNT(*) AS n FROM student_skills WHERE user_id = ? AND mastery >= 4",
            (user_id,),
        ).fetchone()["n"]
        attempts = c.execute(
            "SELECT COUNT(*) AS n FROM student_attempts WHERE user_id = ?", (user_id,)
        ).fetchone()["n"]
    return {
        "skills_tracked": total,
        "avg_mastery": round(avg or 0.0, 2),
        "due_count": due,
        "strong_count": strong,
        "total_attempts": attempts,
    }
