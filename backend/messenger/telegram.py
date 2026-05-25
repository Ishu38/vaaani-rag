"""Telegram Bot API transport.

Webhook receives Update objects from Telegram, normalizes to
IncomingMessage, hands to dispatcher, and sends each OutgoingReply via
the Bot API. v0.1 handles text + audio outbound; document/voice inbound
deferred.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx

from .dispatch import IncomingMessage, OutgoingReply, handle_incoming

log = logging.getLogger("vaaani.messenger.telegram")

TELEGRAM_API = "https://api.telegram.org"
KIND = "telegram"


def _token() -> str | None:
    return (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip() or None


def _webhook_secret() -> str | None:
    """Telegram sends our chosen secret in X-Telegram-Bot-Api-Secret-Token
    on every webhook call when we set one via setWebhook. Keeps the
    webhook URL itself from being a credential."""
    return (os.environ.get("TELEGRAM_WEBHOOK_SECRET") or "").strip() or None


def configured() -> bool:
    return _token() is not None


_BOT_USERNAME_CACHE: str | None = None


def bot_username() -> str | None:
    """Return the bot's @handle so the web UI can render a deep link
    like `https://t.me/<handle>?start=<code>`. Cached after first lookup.
    Returns None if the token isn't configured or Telegram is unreachable."""
    global _BOT_USERNAME_CACHE
    if _BOT_USERNAME_CACHE is not None:
        return _BOT_USERNAME_CACHE or None
    token = _token()
    if not token:
        return None
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{TELEGRAM_API}/bot{token}/getMe")
        if r.status_code == 200:
            data = r.json()
            uname = (data.get("result") or {}).get("username", "")
            _BOT_USERNAME_CACHE = uname
            return uname or None
    except Exception:
        log.exception("getMe failed")
    _BOT_USERNAME_CACHE = ""
    return None


def verify_webhook_secret(header_value: str | None) -> bool:
    """True if no secret is configured (dev) or the header matches."""
    expected = _webhook_secret()
    if not expected:
        return True
    return (header_value or "") == expected


# =========================================================================
#  Inbound (webhook handler)
# =========================================================================

def parse_update(update: dict) -> tuple[IncomingMessage | None, dict | None]:
    """Map a Telegram Update to our normalized form. Returns
    (IncomingMessage, attachment_descriptor) — the descriptor describes
    a file we need to pre-download before invoking the dispatcher.
    Both are None for updates we don't yet handle."""
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return None, None
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return None, None
    text = (msg.get("text") or msg.get("caption") or "").strip()
    sender = msg.get("from") or {}
    username = sender.get("username") or sender.get("first_name")

    incoming = IncomingMessage(
        kind=KIND,
        chat_id=str(chat_id),
        text=text,
        username=username,
    )

    # Photos: Telegram sends an array of progressively larger sizes;
    # pick the last one (biggest) for OCR quality.
    photos = msg.get("photo") or []
    if photos:
        biggest = photos[-1]
        return incoming, {
            "kind": "photo",
            "file_id": biggest.get("file_id"),
            "mime_type": "image/jpeg",
            "filename": None,
        }

    # Documents: PDFs we ingest natively, other types we politely reject
    # in the dispatcher.
    doc = msg.get("document")
    if doc:
        mime = (doc.get("mime_type") or "").lower()
        fname = doc.get("file_name") or ""
        kind = "pdf" if (mime == "application/pdf" or fname.lower().endswith(".pdf")) else "document"
        return incoming, {
            "kind": kind,
            "file_id": doc.get("file_id"),
            "mime_type": mime,
            "filename": fname,
        }

    return incoming, None


def _download_file(file_id: str, suffix: str = "") -> Path | None:
    """Telegram's two-step file fetch: getFile → file_path → CDN URL."""
    token = _token()
    if not token or not file_id:
        return None
    try:
        with httpx.Client(timeout=30.0) as client:
            meta_r = client.get(f"{TELEGRAM_API}/bot{token}/getFile", params={"file_id": file_id})
            if meta_r.status_code != 200:
                log.warning("getFile %s failed: %s", file_id, meta_r.text[:200])
                return None
            file_path = (meta_r.json().get("result") or {}).get("file_path")
            if not file_path:
                return None
            cdn = f"{TELEGRAM_API}/file/bot{token}/{file_path}"
            data_r = client.get(cdn)
            if data_r.status_code != 200:
                log.warning("download %s failed: %s", file_path, data_r.status_code)
                return None
        # Use the original file_path's extension if available.
        ext = Path(file_path).suffix or suffix or ".bin"
        tmp = tempfile.NamedTemporaryFile(prefix="vaaani-tg-", suffix=ext, delete=False)
        tmp.write(data_r.content)
        tmp.close()
        return Path(tmp.name)
    except Exception:
        log.exception("file download failed")
        return None


def handle_update(update: dict) -> list[OutgoingReply]:
    """Top-level webhook handler. Pre-downloads any attachment so the
    dispatcher receives a ready-to-OCR local path."""
    msg, attach = parse_update(update)
    if msg is None:
        return []
    if attach:
        suffix = ".pdf" if attach["kind"] == "pdf" else (".jpg" if attach["kind"] == "photo" else "")
        path = _download_file(attach["file_id"], suffix=suffix)
        if path is not None:
            msg.attachment_path = str(path)
            msg.attachment_kind = attach["kind"]
            msg.attachment_mime = attach.get("mime_type")
            msg.attachment_name = attach.get("filename")
        else:
            return [OutgoingReply(
                kind=KIND, chat_id=msg.chat_id,
                text="I couldn't download that file from Telegram. Try again in a moment.",
            )]
    try:
        return handle_incoming(msg)
    except Exception as e:
        log.exception("dispatch failed")
        return [OutgoingReply(
            kind=KIND,
            chat_id=msg.chat_id,
            text=f"Internal error: {e}",
        )]


# =========================================================================
#  Outbound (send replies)
# =========================================================================

def send_replies(replies: list[OutgoingReply]) -> list[dict]:
    """Send each reply via the Bot API. Returns the raw responses for
    debugging. Failures are logged but don't raise — one failed message
    shouldn't kill the rest of a multi-reply turn."""
    token = _token()
    if not token:
        log.warning("TELEGRAM_BOT_TOKEN not set; dropping %d replies", len(replies))
        return []
    out: list[dict] = []
    with httpx.Client(timeout=30.0) as client:
        for r in replies:
            try:
                if r.audio_url:
                    out.append(_send_audio(client, token, r))
                if r.text:
                    out.append(_send_text(client, token, r))
            except Exception as e:
                log.exception("send failed for chat %s", r.chat_id)
                out.append({"ok": False, "error": str(e)})
    return out


def _send_text(client: httpx.Client, token: str, reply: OutgoingReply) -> dict:
    payload: dict[str, Any] = {
        "chat_id": reply.chat_id,
        "text": reply.text,
        "disable_notification": reply.silent,
        "disable_web_page_preview": True,
    }
    if reply.parse_mode:
        # Telegram's "Markdown" is legacy; "MarkdownV2" is stricter but
        # requires escaping. Stick with legacy Markdown for v0.1 to keep
        # our reply templates simple. Fall back to plain text if Telegram
        # rejects the formatting (common when answers contain stray * or _).
        payload["parse_mode"] = reply.parse_mode
    r = client.post(f"{TELEGRAM_API}/bot{token}/sendMessage", json=payload)
    if r.status_code != 200 and "parse_mode" in payload:
        # Retry without parse_mode so a single stray underscore doesn't
        # eat the whole answer.
        payload.pop("parse_mode")
        r = client.post(f"{TELEGRAM_API}/bot{token}/sendMessage", json=payload)
    return r.json()


def _send_audio(client: httpx.Client, token: str, reply: OutgoingReply) -> dict:
    """Send via Telegram's sendAudio. We pass the audio_url so Telegram
    fetches it server-to-server — works because brain.vaaani.in's
    /audio/file/<hash>.mp3 endpoint is public-readable."""
    payload: dict[str, Any] = {
        "chat_id": reply.chat_id,
        "audio": reply.audio_url,
        "disable_notification": reply.silent,
    }
    if reply.text:
        payload["caption"] = reply.text
        if reply.parse_mode:
            payload["parse_mode"] = reply.parse_mode
    r = client.post(f"{TELEGRAM_API}/bot{token}/sendAudio", json=payload)
    return r.json()


# =========================================================================
#  Webhook registration helper (manual one-off)
# =========================================================================

def register_webhook(public_url: str) -> dict:
    """Tell Telegram where our webhook lives. Called manually after
    deploy — wire this into a make-target or invoke once per env."""
    token = _token()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
    payload = {
        "url": public_url,
        "allowed_updates": ["message", "edited_message"],
    }
    secret = _webhook_secret()
    if secret:
        payload["secret_token"] = secret
    with httpx.Client(timeout=15.0) as client:
        r = client.post(f"{TELEGRAM_API}/bot{token}/setWebhook", json=payload)
    return r.json()
