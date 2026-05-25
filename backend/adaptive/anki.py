"""Anki .apkg export of Vaaani's review queue.

Generates a single deck per request containing one cloze card per graph
node the user has reviewable cloze material for, plus optional basic
recall cards for nodes that lack a verbatim cloze. The deck preserves
Vaaani's source attribution (filename → tag + footer) so the student
can still trace any card back to its passage.

Why .apkg and not CSV:
  - Anki's native package keeps deck name, card model, CSS, and the
    cloze/basic distinction in one file. CSV would force the user to
    pre-build a matching deck + note type in Anki first.

Why this leaves graph-aware interleaving on the Vaaani side (not Anki):
  - Anki picks next-card by due-date alone. Our moat is interleaving by
    graph distance. Cards exported to Anki return to vanilla scheduling
    — that's an acceptable v0.1 trade. A custom Anki note-type or addon
    could re-implement the graph signal inside Anki later; out of
    scope here.
"""
from __future__ import annotations

import io
import re
import tempfile
from datetime import datetime
from pathlib import Path

import genanki

from auth.db import connect
from . import spaced


# Stable model ids (random int per https://github.com/kerrickstaley/genanki#model)
_CLOZE_MODEL_ID = 1872194501
_BASIC_MODEL_ID = 1872194502


_CARD_CSS = """
.card {
  font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
  font-size: 18px; line-height: 1.55;
  color: #3c5340;
  background: #fff3d5;
  padding: 30px 24px;
  text-align: left;
}
.cloze { color: #a06a1e; font-weight: 700; }
.vaaani-src { color: #7a8b6f; font-size: 12.5px; margin-top: 18px; border-top: 1px solid #e0cc93; padding-top: 8px; }
.vaaani-src b { color: #5d7459; font-weight: 600; }
.vaaani-desc { color: #5d7459; font-size: 14px; margin-top: 14px; }
"""


_CLOZE_MODEL = genanki.Model(
    _CLOZE_MODEL_ID,
    "Vaaani Cloze",
    fields=[
        {"name": "Text"},
        {"name": "Source"},
        {"name": "Description"},
    ],
    templates=[
        {
            "name": "Vaaani cloze card",
            "qfmt": "{{cloze:Text}}<div class='vaaani-src'>From <b>{{Source}}</b></div>",
            "afmt": "{{cloze:Text}}<div class='vaaani-desc'>{{Description}}</div><div class='vaaani-src'>From <b>{{Source}}</b></div>",
        },
    ],
    css=_CARD_CSS,
    model_type=genanki.Model.CLOZE,
)

_BASIC_MODEL = genanki.Model(
    _BASIC_MODEL_ID,
    "Vaaani Recall",
    fields=[
        {"name": "Front"},
        {"name": "Back"},
        {"name": "Source"},
    ],
    templates=[
        {
            "name": "Vaaani recall card",
            "qfmt": "{{Front}}",
            "afmt": "{{FrontSide}}<hr id='answer'>{{Back}}<div class='vaaani-src'>{{#Source}}From <b>{{Source}}</b>{{/Source}}</div>",
        },
    ],
    css=_CARD_CSS,
)


_TAG_RE = re.compile(r"[^A-Za-z0-9]+")


def _safe_tag(s: str) -> str:
    """Anki tags must not contain spaces; lowercase + collapse non-alnum."""
    s = _TAG_RE.sub("_", s.strip().lower())
    return s.strip("_") or "untagged"


def _to_anki_cloze(raw: str, display: str) -> str | None:
    """Replace the first case-insensitive occurrence of `display` in `raw`
    with Anki's {{c1::...}} syntax, preserving the matched casing.

    Returns None if the term isn't found verbatim (shouldn't happen since
    raw was selected for containing the term, but defensive).
    """
    pattern = re.compile(rf"\b{re.escape(display)}\b", re.IGNORECASE)
    match = pattern.search(raw)
    if not match:
        return None
    matched_text = match.group(0)
    return raw[:match.start()] + "{{c1::" + matched_text + "}}" + raw[match.end():]


def _gather_user_nodes(user_id: int, limit: int = 500) -> list[dict]:
    """Pull the user's tracked nodes (highest mastery → lowest). Falls
    back to top-degree corpus nodes when the user has no tracked state
    yet, so first-time exports still produce a usable starter deck."""
    with connect() as c:
        rows = c.execute(
            """SELECT topic, display, mastery FROM student_skills
                WHERE user_id = ?
                ORDER BY mastery DESC, attempts DESC
                LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    if rows:
        return [{"topic": r["topic"], "display": r["display"], "mastery": float(r["mastery"])} for r in rows]
    # No tracked state — seed from the corpus graph by node degree.
    nodes, adj = spaced._graph()
    by_degree = sorted(nodes.items(), key=lambda kv: -len(adj.get(kv[0], ())))
    out: list[dict] = []
    for nid, n in by_degree[:limit]:
        out.append({"topic": nid, "display": n.get("display", nid), "mastery": 2.0})
    return out


def _build_deck_name() -> str:
    return f"Vaaani Study Pack · {datetime.utcnow().strftime('%Y-%m-%d')}"


def build_apkg_for_user(user_id: int) -> tuple[bytes, str, dict]:
    """Build an .apkg containing one card per node with usable material.
    Returns (apkg_bytes, suggested_filename, stats_dict)."""
    nodes = _gather_user_nodes(user_id)
    if not nodes:
        raise ValueError("no nodes in graph yet — ingest at least one document")

    graph_nodes, _ = spaced._graph()
    deck_name = _build_deck_name()
    deck = genanki.Deck(
        deck_id=abs(hash(("vaaani", user_id, deck_name))) % (1 << 31),
        name=deck_name,
    )

    n_cloze = 0
    n_recall = 0
    n_skipped = 0

    for entry in nodes:
        node_id = entry["topic"]
        display = entry["display"]
        node_meta = graph_nodes.get(node_id, {})
        node_type = node_meta.get("type", "")
        descriptions = node_meta.get("descriptions") or []
        description = (descriptions[0] if descriptions else "").strip()

        tags = ["vaaani"]
        if node_type:
            tags.append(_safe_tag(node_type))

        # Try a cloze first; if the corpus has no verbatim sentence,
        # build a recall card instead.
        cloze = spaced._find_cloze_passage(display)
        if cloze:
            raw, _placeholder, src = cloze
            anki_text = _to_anki_cloze(raw, display)
            if anki_text is None:
                n_skipped += 1
                continue
            deck.add_note(genanki.Note(
                model=_CLOZE_MODEL,
                fields=[anki_text, src or "", description or ""],
                tags=tags,
                guid=genanki.guid_for(user_id, node_id, "cloze"),
            ))
            n_cloze += 1
            continue

        # Recall fallback: only generate when the node has at least
        # one neighbour and a description, otherwise the card is empty.
        if not description:
            n_skipped += 1
            continue
        deck.add_note(genanki.Note(
            model=_BASIC_MODEL,
            fields=[
                f"Explain <b>{display}</b> in your own words.",
                description,
                "",
            ],
            tags=tags,
            guid=genanki.guid_for(user_id, node_id, "recall"),
        ))
        n_recall += 1

    if not (n_cloze + n_recall):
        raise ValueError("no usable cards — ingest more material so the graph has cloze-able sentences")

    package = genanki.Package(deck)
    with tempfile.NamedTemporaryFile(suffix=".apkg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        package.write_to_file(str(tmp_path))
        data = tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)

    filename = f"vaaani-study-{datetime.utcnow().strftime('%Y%m%d')}.apkg"
    stats = {
        "deck_name": deck_name,
        "cards": n_cloze + n_recall,
        "cloze_cards": n_cloze,
        "recall_cards": n_recall,
        "skipped": n_skipped,
        "bytes": len(data),
    }
    return data, filename, stats
