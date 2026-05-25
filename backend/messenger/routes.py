"""FastAPI routes for Telegram (and future WhatsApp) — webhook + link flow.

Web-side endpoints (auth-gated, cookie session):
  POST /messenger/link/start    → mint a one-time code the user pastes in the bot
  POST /messenger/unlink        → remove all bot links for the signed-in user
  GET  /messenger/status        → show what's linked

Transport webhook (called by the bot platform, secured by header secret):
  POST /messenger/telegram/webhook
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Cookie, Header, HTTPException, Request
from pydantic import BaseModel, Field

from auth import service as auth_service
from auth.security import decode_session
from config import COOKIE_NAME

from . import store
from . import telegram as tg

router = APIRouter(prefix="/messenger", tags=["messenger"])


def _require_user(session_cookie: str | None) -> dict:
    payload = decode_session(session_cookie or "")
    if not payload:
        raise HTTPException(401, "Not signed in.")
    user = auth_service.get_user_by_id(int(payload["sub"]))
    if not user:
        raise HTTPException(401, "Not signed in.")
    return user


# ---------------- web-side (account linking) ----------------

class LinkStartBody(BaseModel):
    kind: str = Field(..., pattern="^(telegram|whatsapp)$")


@router.post("/link/start")
def link_start(
    body: LinkStartBody,
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """Mint a short-lived code the user types in the bot to connect."""
    user = _require_user(vaaani_session)
    code = store.mint_link_code(user["id"], body.kind)
    return {
        "code": code,
        "kind": body.kind,
        "expires_in_minutes": store.LINK_CODE_TTL_MIN,
        "bot_username": tg.bot_username() if body.kind == "telegram" else None,
    }


class UnlinkBody(BaseModel):
    kind: Optional[str] = Field(None, pattern="^(telegram|whatsapp)$")


@router.post("/unlink")
def unlink(
    body: UnlinkBody = UnlinkBody(),
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    user = _require_user(vaaani_session)
    removed = store.unlink_user(user["id"], body.kind)
    return {"removed": removed}


@router.get("/status")
def status(vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME)):
    user = _require_user(vaaani_session)
    links = store.list_links_for_user(user["id"])
    return {
        "links": links,
        "telegram_configured": tg.configured(),
        "telegram_bot_username": tg.bot_username(),
    }


@router.get("/public")
def public_status():
    """Unauthenticated metadata: which messengers are wired on this
    deployment. Used by the Integrations panel so visitors see honest
    badges before they sign in."""
    return {
        "telegram_configured": tg.configured(),
        "telegram_bot_username": tg.bot_username(),
    }


# ---------------- bot-side (webhook) ----------------

@router.post("/telegram/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None),
):
    """Telegram POSTs an Update here on every message. Returns 200 quickly
    (Telegram retries on non-2xx, which we don't want amplifying a slow
    DeepSeek call)."""
    if not tg.verify_webhook_secret(x_telegram_bot_api_secret_token):
        raise HTTPException(403, "bad secret")
    payload = await request.json()
    replies = tg.handle_update(payload)
    tg.send_replies(replies)
    return {"ok": True}
