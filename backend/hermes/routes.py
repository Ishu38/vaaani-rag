"""FastAPI router for /hermes/* inspection endpoints.

Read-only. The corrector itself runs inline inside /chat; these routes just
let the user see what Hermes has learned.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Cookie

from auth import service as auth_service
from auth.security import decode_session
from config import COOKIE_NAME

from hermes import patterns, store

router = APIRouter(prefix="/hermes", tags=["hermes"])


def _maybe_user_id(session_cookie: str | None) -> int | None:
    """Resolve user_id from cookie if present; otherwise scope to anonymous traces."""
    payload = decode_session(session_cookie or "")
    if not payload:
        return None
    try:
        u = auth_service.get_user_by_id(int(payload["sub"]))
        return int(u["id"]) if u else None
    except (KeyError, ValueError, TypeError):
        return None


@router.get("/stats")
def stats(vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME)):
    """Top-level Hermes dashboard: how much it's seen, how often it's failing/correcting."""
    uid = _maybe_user_id(vaaani_session)
    return {
        "scope": "user" if uid else "anonymous",
        "overall": patterns.overall_stats(uid),
        "correction_effectiveness": patterns.correction_effectiveness(uid),
    }


@router.get("/recent")
def recent(
    limit: int = 25,
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """Last N traces (no embeddings) for inspection. Capped at 100."""
    uid = _maybe_user_id(vaaani_session)
    return {"traces": store.recent_traces(uid, limit=min(max(1, limit), 100))}


@router.get("/patterns")
def patterns_endpoint(vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME)):
    """Weak query templates Hermes is targeting for self-correction."""
    uid = _maybe_user_id(vaaani_session)
    return {"weak_templates": patterns.weak_query_templates(uid)}
