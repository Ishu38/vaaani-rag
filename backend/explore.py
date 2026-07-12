"""Explore My World — the camera-led discovery loop.

A child points the camera at something real; the object is recognised (client
side), and Vaaani walks the learner through the Language Journey for that
object — Recognize -> Construct -> Reason -> Communicate — ending with the
child recording a short video *telling the object's story* (the Communicate
capstone: producing language, the anti-reel). Each completed step grows the
object into a brighter star in the child's universe.

This module owns the per-learner discovery store and the deterministic,
grade-appropriate Journey scaffold. The conversation is templated (instant,
pedagogically controlled, and safe for young children) — the adaptive tutor
engine can later enrich the "Communicate" feedback; the scaffold stays the
spine so a slow model never leaves a 6-year-old waiting.

Privacy: discoveries and any video live under the child's own scope (see
scope.py). A child's camera content never becomes another learner's data.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from config import DATA_DIR

EXPLORE_DB = DATA_DIR / "explore.db"
EXPLORE_MEDIA = DATA_DIR / "explore_media"
EXPLORE_MEDIA.mkdir(parents=True, exist_ok=True)

# The Language Journey, as a concrete per-object scaffold. Each step is one
# gentle question; answering it advances the child and brightens the star.
# No linguistic notation here by design — this is vocabulary and description,
# the Foundational-stage oral work that precedes any symbol.
JOURNEY_STEPS = [
    ("recognize", "Recognize", "You found {a} {obj}! Say its name out loud — {obj}. What is it?"),
    ("colour",    "Construct", "What colour is your {obj}?"),
    ("use",       "Construct", "What do we use {a} {obj} for?"),
    ("where",     "Reason",    "Where do we usually find {a} {obj}?"),
    ("compare",   "Reason",    "Can you think of something a bit like {a} {obj}? How are they different?"),
    ("story",     "Communicate", "Now the best part — tell a little story about your {obj}. "
                                  "Tap record and say it out loud!"),
]
N_STEPS = len(JOURNEY_STEPS)


def _article(word: str) -> str:
    """'a' / 'an' by sound-ish rule — a language tutor must not say 'a apple'."""
    return "an" if (word[:1].lower() in "aeiou") else "a"


def _conn() -> sqlite3.Connection:
    db = sqlite3.connect(str(EXPLORE_DB))
    db.row_factory = sqlite3.Row
    db.execute("""
        CREATE TABLE IF NOT EXISTS explore_discoveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            object TEXT NOT NULL,
            step INTEGER NOT NULL DEFAULT 0,          -- Journey steps completed
            answers TEXT NOT NULL DEFAULT '',         -- child's answers, newline-joined
            video_path TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(user_id, object)
        )
    """)
    db.execute("CREATE INDEX IF NOT EXISTS ix_explore_user ON explore_discoveries(user_id)")
    # point-see-say grounding columns (added incrementally; ignore if present)
    for col, decl in [("vision_label", "TEXT"), ("spoken_label", "TEXT"),
                      ("pointing", "INTEGER DEFAULT 0"), ("grounding", "TEXT")]:
        try:
            db.execute(f"ALTER TABLE explore_discoveries ADD COLUMN {col} {decl}")
        except sqlite3.OperationalError:
            pass  # column already exists
    return db


def _mastery_from_step(step: int) -> float:
    """A star brightens as the child completes the Journey — full brightness
    (5/5) once all six steps are done."""
    return round(min(5.0, 5.0 * step / N_STEPS), 2)


def _norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def fuse(vision_label: str, spoken_label: str, pointing: bool) -> dict:
    """POINT-SEE-SAY grounding — the multimodal core.

    A human child grounds a word in three signals at once: what they SEE, what
    they SAY, and what they POINT at. Fusing them is what turns a lookup into
    acquisition, and it is the defensible, uniquely-ours mechanism (it needs the
    vision engine AND the gesture engine we own). Agreement across signals is
    strong grounding; disagreement is not an error but a teachable moment.

    Returns the grounded object and a grounding descriptor:
      - see+say (+point): vision and speech agree → strongest grounding
      - mismatch: vision and speech differ → keep the child's spoken word as the
        target, flag what the camera saw so the tutor can gently reconcile
      - see-only / say-only: one signal present → partial grounding
    """
    v, s = _norm(vision_label), _norm(spoken_label)
    if v and s and v == s:
        conf = "high"
        mode = "see+say+point" if pointing else "see+say"
        return {"object": v, "grounded": True, "grounding": mode, "confidence": conf,
                "vision_saw": v, "spoken": s, "pointing": bool(pointing)}
    if v and s and v != s:
        # trust the child's word as the object; surface the mismatch for the tutor
        return {"object": s, "grounded": False, "grounding": "mismatch", "confidence": "low",
                "vision_saw": v, "spoken": s, "pointing": bool(pointing)}
    obj = s or v
    return {"object": obj, "grounded": bool(obj),
            "grounding": ("say" if s else "see") if obj else "none",
            "confidence": "medium" if obj else "none",
            "vision_saw": v, "spoken": s, "pointing": bool(pointing)}


def ground(user_id: int, vision_label: str = "", spoken_label: str = "",
           pointing: bool = False) -> dict:
    """Fuse the three signals and begin (or re-open) the discovery for the
    grounded object. Returns Journey state enriched with the grounding."""
    f = fuse(vision_label, spoken_label, pointing)
    obj = f["object"]
    if not obj:
        raise ValueError("no signal — point, see or say something")
    state = start_discovery(user_id, obj, _grounding=f)
    state["grounding"] = f["grounding"]
    state["confidence"] = f["confidence"]
    state["vision_saw"] = f["vision_saw"]
    state["spoken"] = f["spoken"]
    state["pointing"] = f["pointing"]
    return state


def start_discovery(user_id: int, obj: str, _grounding: dict | None = None) -> dict:
    """Register (or re-open) a discovery for this object. Returns the current
    Journey state and the next question."""
    obj = " ".join((obj or "").split()).lower()
    if not obj:
        raise ValueError("empty object")
    g = _grounding or {}
    with _conn() as db:
        db.execute(
            "INSERT INTO explore_discoveries (user_id, object, vision_label, spoken_label, pointing, grounding) "
            "VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(user_id, object) DO UPDATE SET updated_at=datetime('now'), "
            "  vision_label=COALESCE(excluded.vision_label, vision_label), "
            "  spoken_label=COALESCE(excluded.spoken_label, spoken_label), "
            "  pointing=MAX(pointing, excluded.pointing), "
            "  grounding=COALESCE(excluded.grounding, grounding)",
            (user_id, obj, g.get("vision_saw") or None, g.get("spoken") or None,
             1 if g.get("pointing") else 0, g.get("grounding")),
        )
        row = db.execute(
            "SELECT * FROM explore_discoveries WHERE user_id=? AND object=?",
            (user_id, obj),
        ).fetchone()
    return _state(row)


def answer_step(user_id: int, obj: str, answer: str) -> dict:
    """Record the child's answer to the current step and advance."""
    obj = " ".join((obj or "").split()).lower()
    with _conn() as db:
        row = db.execute(
            "SELECT * FROM explore_discoveries WHERE user_id=? AND object=?",
            (user_id, obj),
        ).fetchone()
        if not row:
            raise ValueError("no such discovery")
        step = min(row["step"] + 1, N_STEPS)
        answers = (row["answers"] + "\n" + (answer or "").strip()).strip()
        db.execute(
            "UPDATE explore_discoveries SET step=?, answers=?, updated_at=datetime('now') "
            "WHERE id=?",
            (step, answers, row["id"]),
        )
        row = db.execute("SELECT * FROM explore_discoveries WHERE id=?", (row["id"],)).fetchone()
    return _state(row)


def attach_video(user_id: int, obj: str, filename: str) -> dict:
    obj = " ".join((obj or "").split()).lower()
    with _conn() as db:
        db.execute(
            "UPDATE explore_discoveries SET video_path=?, step=?, updated_at=datetime('now') "
            "WHERE user_id=? AND object=?",
            (filename, N_STEPS, user_id, obj),
        )
        row = db.execute(
            "SELECT * FROM explore_discoveries WHERE user_id=? AND object=?",
            (user_id, obj),
        ).fetchone()
    return _state(row)


def list_discoveries(user_id: int) -> list[dict]:
    with _conn() as db:
        rows = db.execute(
            "SELECT * FROM explore_discoveries WHERE user_id=? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return [_state(r) for r in rows]


def _state(row: sqlite3.Row) -> dict:
    step = row["step"]
    done = step >= N_STEPS
    if done:
        question = f"You told the whole story of your {row['object']}! It's a bright star now. Explore something new!"
        stage = "Communicate"
    else:
        key, stage, template = JOURNEY_STEPS[step]
        question = template.format(obj=row["object"], a=_article(row["object"]))
    return {
        "id": row["id"],
        "object": row["object"],
        "step": step,
        "total_steps": N_STEPS,
        "stage": stage,
        "question": question,
        "done": done,
        "mastery": _mastery_from_step(step),
        "has_video": bool(row["video_path"]),
    }


# ── Camera → Knowledge Graph theme matching ──────────────────────────
# When the camera recognises an object, we look up related word-family
# themes in the knowledge graph so the child can go from "I saw a person"
# → "Let me explore all the people words!" — bridging real-world
# perception to the structured language universe.

# Direct map: COCO-SSD class name → graph community IDs
# These are curated so the camera always finds the right word family.
COCO_TO_COMMUNITY: dict[str, str] = {
    # People / family objects → person family communities
    "person":    "person",
    "man":       "person",
    "woman":     "person",
    "boy":       "person",
    "girl":      "person",
    # Animals → animal communities
    "cat":       "cat",
    "dog":       "dog",
    "bird":      "dog",     # closest match: pets
    "horse":     "dog",
    "cow":       "dog",
    "elephant":  "dog",
    "bear":      "dog",
    "zebra":     "dog",
    "giraffe":   "dog",
    "sheep":     "dog",
    "teddy bear":"dog",
    # Furniture / home
    "chair":     "chair",
    "couch":     "chair",
    "bench":     "chair",
    "dining table": "table",
    "bed":       "home",
    "toilet":    "home",
    "potted plant": "home",
    # Objects
    "book":      "book",
    "cup":       "cup",
    "bottle":    "bottle",
    "wine glass":"cup",
    "spoon":     "cup",     # closest: kitchen items
    "fork":      "cup",
    "knife":     "cup",
    "bowl":      "cup",
    # Food → food community
    "apple":     "food",
    "banana":    "food",
    "sandwich":  "food",
    "orange":    "food",
    "broccoli":  "food",
    "carrot":    "food",
    "hot dog":   "food",
    "pizza":     "food",
    "donut":     "food",
    "cake":      "food",
    # Transport → vehicle community
    "bicycle":   "vehicle",
    "car":       "vehicle",
    "motorcycle":"vehicle",
    "airplane":  "vehicle",
    "bus":       "vehicle",
    "train":     "vehicle",
    "truck":     "vehicle",
    "boat":      "vehicle",
    # Electronics → home (for now; future: technology community)
    "laptop":    "home",
    "mouse":     "home",
    "keyboard":  "home",
    "cell phone": "home",
    "tv":        "home",
    "remote":    "home",
    # Clothing / accessories
    "backpack":  "home",
    "umbrella":  "home",
    "handbag":   "home",
    "tie":       "home",
    "suitcase":  "home",
    "sports ball": "home",
    # Other
    "clock":     "home",
    "vase":      "home",
    "scissors":  "home",
    "toothbrush":"home",
    "hair drier":"home",
}


def match_themes(obj: str, user_id: int | None = None) -> dict:
    """Search the knowledge graph for themes connected to a detected object.

    Returns a list of matching communities with their nodes and summaries,
    plus any directly-matched entities from the graph.
    """
    import json
    from config import GRAPH_PATH, COMMUNITIES_PATH
    from community import Community

    obj_norm = " ".join((obj or "").split()).lower()
    if not obj_norm:
        return {"object": obj, "themes": [], "entities": []}

    # ── Load graph ──
    entities: list[dict] = []
    if GRAPH_PATH.exists():
        with open(GRAPH_PATH) as f:
            g = json.load(f)
        for n in g.get("nodes", []):
            disp = (n.get("display") or "").lower()
            if not disp:
                continue
            # Fuzzy: exact contains or word overlap
            score = 0
            if disp == obj_norm:
                score = 3
            elif obj_norm in disp or disp in obj_norm:
                score = 2
            else:
                # Word overlap
                obj_words = set(obj_norm.split())
                disp_words = set(disp.split())
                overlap = obj_words & disp_words
                if overlap:
                    score = min(1, len(overlap) / max(len(obj_words), len(disp_words)))

            if score > 0:
                entities.append({
                    "name": n.get("display", n.get("id", "")),
                    "type": n.get("type", "unknown"),
                    "score": score,
                    "descriptions": n.get("descriptions", [])[:2],
                })

    entities.sort(key=lambda e: -e["score"])

    # ── Load communities ──
    themes: list[dict] = []
    if COMMUNITIES_PATH.exists():
        with open(COMMUNITIES_PATH) as f:
            raw = json.load(f)
        for c in raw:
            match_reason = []
            best_score = 0
            for n in c.get("nodes", []):
                for e in entities:
                    if n.lower() == e["name"].lower():
                        match_reason.append(n)
                        if e["score"] > best_score:
                            best_score = e["score"]
            if match_reason:
                themes.append({
                    "id": c["id"],
                    "title": c.get("title", ""),
                    "summary": c.get("summary", ""),
                    "findings": c.get("findings", [])[:3],
                    "nodes": c.get("nodes", [])[:8],
                    "size": c.get("size", len(c.get("nodes", []))),
                    "matched_via": match_reason[:3],
                    "best_match_score": best_score,
                })

    # Sort: exact entity matches first, then person/family themes, then by size
    person_keywords = {"person", "father", "mother", "parent", "child", "family",
                       "brother", "sister", "friend", "neighbor", "man", "woman", "boy", "girl"}
    def _priority(t):
        best = t.get("best_match_score", 0)
        is_person = any(kw in t["title"].lower() for kw in person_keywords)
        return (-best, not is_person, -t["size"])
    themes.sort(key=_priority)

    # ── Curated fallback via COCO mapping ──────────────────────────
    # If no themes were found through text matching, look up the
    # community by the curated COCO→community name mapping.
    if not themes and obj_norm in COCO_TO_COMMUNITY:
        target_name = COCO_TO_COMMUNITY[obj_norm].lower()
        if COMMUNITIES_PATH.exists():
            with open(COMMUNITIES_PATH) as f:
                raw = json.load(f)
            for c in raw:
                if target_name in c.get("title", "").lower() or target_name == c.get("title", "").lower().split("—")[0].strip().lower():
                    themes.append({
                        "id": c["id"],
                        "title": c.get("title", ""),
                        "summary": c.get("summary", ""),
                        "findings": c.get("findings", [])[:3],
                        "nodes": c.get("nodes", [])[:8],
                        "size": c.get("size", len(c.get("nodes", []))),
                        "matched_via": [target_name],
                    })
                    break

    return {
        "object": obj,
        "themes": themes[:8],
        "entities": entities[:8],
    }
