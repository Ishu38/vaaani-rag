"""FastAPI routes for YouTube transcript ingest.

Auth-gated. One endpoint:

  POST /youtube/ingest  { "url": "<youtube URL or video id>" }

We fetch the transcript via youtube-transcript-api (no API key, public
timed-text endpoint), flatten it to a markdown doc with timestamped
paragraphs, drop it into data/raw/ as `youtube-<id>-<ts>.md`, and run
the existing ingest pipeline. Re-ingest of the same video id replaces
the prior file (stable filename keyed by video id).
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Cookie, HTTPException
from pydantic import BaseModel, Field

from auth import service as auth_service
from auth.security import decode_session
from config import COOKIE_NAME, INDEX_PATH, METADATA_PATH, RAW_DIR
from ingest import ingest

from . import client as yt_client

router = APIRouter(prefix="/youtube", tags=["youtube"])


def _require_user(session_cookie: str | None) -> dict:
    payload = decode_session(session_cookie or "")
    if not payload:
        raise HTTPException(401, "Not signed in.")
    user = auth_service.get_user_by_id(int(payload["sub"]))
    if not user:
        raise HTTPException(401, "Not signed in.")
    return user


class IngestBody(BaseModel):
    url: str = Field(..., min_length=1, max_length=400)


_SAFE_ID = re.compile(r"[A-Za-z0-9_-]{11}")


@router.post("/ingest")
def ingest_video(
    body: IngestBody,
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    yt_user = _require_user(vaaani_session)
    try:
        video_id = yt_client.parse_video_id(body.url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not _SAFE_ID.fullmatch(video_id):
        # Belt and braces — parse_video_id only returns 11-char ids,
        # but we'll be the file path's last line of defence anyway.
        raise HTTPException(400, "invalid video id")
    try:
        title, body_md = yt_client.fetch_transcript_markdown(video_id)
    except yt_client.TranscriptError as e:
        raise HTTPException(422, str(e))
    if not body_md.strip():
        raise HTTPException(422, "transcript came back empty")

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    fname = f"youtube-{video_id}-{ts}.md"
    user_dir = RAW_DIR / f"u{yt_user['id']}"
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / fname
    front = f"# YouTube transcript · {video_id}\n\n{title}\n\nhttps://youtu.be/{video_id}\n\n"
    path.write_text(front + body_md, encoding="utf-8")

    summary = ingest(RAW_DIR, INDEX_PATH, METADATA_PATH)
    import scope
    scope.record_ownership(
        str(path.resolve()), yt_user["id"], scope.sharing_school_ids(yt_user)
    )
    # Bust caches so the new video shows up in chat / Review / Feynman
    # without a process restart.
    try:
        from main import retriever as _retriever
        _retriever.reload()
    except Exception:
        pass
    try:
        from adaptive import spaced as _spaced
        _spaced.invalidate_graph_cache()
        _spaced.invalidate_chunks_cache()
    except Exception:
        pass

    return {
        "status": "ok",
        "video_id": video_id,
        "filename": fname,
        "chars": len(body_md),
        "chunks_added": summary.get("chunks_added", 0),
        "total_chunks": summary.get("total_chunks", 0),
    }
