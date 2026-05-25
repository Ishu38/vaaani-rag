"""FastAPI router for /learning/* endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Cookie, HTTPException, Response
from pydantic import BaseModel, Field

from auth import service as auth_service
from auth.security import decode_session
from config import COOKIE_NAME

from . import anki as anki_export
from . import service as learn
from . import spaced

router = APIRouter(prefix="/learning", tags=["learning"])


# ---------------- helpers ----------------

class TopicRef(BaseModel):
    """A single (key, display) topic reference."""
    topic: str = Field(..., min_length=1)
    display: str = ""


class RateBody(BaseModel):
    """POST /learning/rate body."""
    rating: int = Field(..., ge=-1, le=1)
    topics: list[TopicRef] = Field(default_factory=list)
    query: str = ""


def _require_user(session_cookie: str | None) -> dict:
    """401-or-user resolver mirroring auth.routes (avoids the import cycle)."""
    payload = decode_session(session_cookie or "")
    if not payload:
        raise HTTPException(status_code=401, detail="Not signed in.")
    user = auth_service.get_user_by_id(int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=401, detail="Not signed in.")
    return user


# ---------------- routes ----------------

@router.post("/rate")
def rate(body: RateBody, vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME)):
    """Record a student rating across one or more topics. Returns the updated rows."""
    user = _require_user(vaaani_session)
    pairs = [(t.topic, t.display) for t in body.topics if (t.topic or t.display)]
    if not pairs:
        return {"updated": []}
    updated = learn.record_attempts_bulk(user["id"], pairs, body.rating, body.query)
    return {"updated": updated}


@router.get("/skills")
def skills(vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME)):
    """Full skill list for the signed-in user."""
    user = _require_user(vaaani_session)
    return {
        "stats": learn.stats(user["id"]),
        "skills": learn.list_skills(user["id"]),
        "due": learn.due_for_review(user["id"]),
    }


# ---------------- spaced review (graph-aware) ----------------

class GradeBody(BaseModel):
    """POST /learning/review/grade body."""
    node_id: str = Field(..., min_length=1)
    display: str = ""
    grade: str = Field(..., pattern="^(again|hard|good|easy)$")


@router.get("/review/next")
def review_next(vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME)):
    """Pull the next due review item, chosen by graph-distance interleaving."""
    user = _require_user(vaaani_session)
    item = spaced.next_review(user["id"])
    return {
        "stats": spaced.session_stats(user["id"]),
        "item": item,
    }


@router.post("/review/grade")
def review_grade(
    body: GradeBody,
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """Apply a grade to the current card and return the next one in a single round-trip."""
    user = _require_user(vaaani_session)
    try:
        updated = spaced.grade_node(user["id"], body.node_id, body.display, body.grade)
    except ValueError as e:
        raise HTTPException(400, str(e))
    nxt = spaced.next_review(user["id"])
    return {
        "graded": updated,
        "stats": spaced.session_stats(user["id"]),
        "item": nxt,
    }


# ---------------- Anki .apkg export ----------------

@router.get("/anki/preview")
def anki_preview(vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME)):
    """Cheap pre-flight: how many cards would the export contain? Used by
    the UI to show a count before the user clicks Download."""
    user = _require_user(vaaani_session)
    nodes = anki_export._gather_user_nodes(user["id"])
    return {
        "candidate_nodes": len(nodes),
        "deck_name": anki_export._build_deck_name(),
    }


@router.get("/anki/export")
def anki_export_apkg(vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME)):
    """Build + stream an .apkg file the user can import into Anki."""
    user = _require_user(vaaani_session)
    try:
        data, filename, stats = anki_export.build_apkg_for_user(user["id"])
    except ValueError as e:
        raise HTTPException(400, str(e))
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        # Surface the counts as response headers so the frontend can show
        # "exported N cards" without parsing the binary body.
        "X-Vaaani-Cards": str(stats["cards"]),
        "X-Vaaani-Cloze-Cards": str(stats["cloze_cards"]),
        "X-Vaaani-Recall-Cards": str(stats["recall_cards"]),
        "X-Vaaani-Skipped": str(stats["skipped"]),
    }
    return Response(content=data, media_type="application/octet-stream", headers=headers)
