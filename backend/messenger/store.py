"""SQLite persistence for messenger links + state + link codes.

Three tables, all keyed by (kind, chat_id) so the same schema serves
Telegram, WhatsApp, and any future transport (Discord, Slack).

  messenger_links     — chat_id ↔ user_id (the "linked account" join)
  messenger_state     — per-chat conversation state (mode + payload)
  messenger_link_codes — short-lived one-time codes minted on the web,
                        consumed by the bot when the user pastes them.
"""
from __future__ import annotations

import secrets
import string
from datetime import datetime, timedelta

from auth.db import connect


# How long a freshly-minted link code stays valid before the user must
# request a new one. 10 minutes balances UX (copy from web, paste in bot)
# against drift from leaked codes.
LINK_CODE_TTL_MIN = 10

# How long messenger_state rows survive without activity. After this,
# the next message resets the user to default chat mode.
STATE_TTL_MIN = 60


def _ensure_tables() -> None:
    with connect() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS messenger_links (
                kind TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                username TEXT,
                linked_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (kind, chat_id)
            );
            CREATE INDEX IF NOT EXISTS ix_mlinks_user ON messenger_links(user_id);

            CREATE TABLE IF NOT EXISTS messenger_state (
                kind TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'chat',
                payload TEXT,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (kind, chat_id)
            );

            CREATE TABLE IF NOT EXISTS messenger_link_codes (
                code TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                kind TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used_at TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS ix_mlc_user ON messenger_link_codes(user_id);

            CREATE TABLE IF NOT EXISTS messenger_prefs (
                kind TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (kind, chat_id, key)
            );
            """
        )


# =========================================================================
#  Link codes (mint on web → consume in bot)
# =========================================================================

def mint_link_code(user_id: int, kind: str) -> str:
    """Generate a short, human-readable, one-time code."""
    _ensure_tables()
    # 8 chars of [a-z0-9], prefixed for legibility in chat. Roughly 40 bits
    # of entropy — fine for a 10-minute single-use code.
    alphabet = string.ascii_lowercase + string.digits
    suffix = "".join(secrets.choice(alphabet) for _ in range(8))
    code = f"vaaani-{suffix}"
    expires = (datetime.utcnow() + timedelta(minutes=LINK_CODE_TTL_MIN)).isoformat()
    with connect() as c:
        c.execute(
            "INSERT INTO messenger_link_codes (code, user_id, kind, expires_at) VALUES (?,?,?,?)",
            (code, user_id, kind, expires),
        )
    return code


def consume_link_code(code: str, kind: str) -> int | None:
    """Returns user_id if the code is valid, unused, and not expired.
    Marks the code consumed atomically. Returns None on any failure
    so the bot can show a generic "invalid or expired" message
    without leaking which case."""
    _ensure_tables()
    now = datetime.utcnow().isoformat()
    with connect() as c:
        row = c.execute(
            """SELECT user_id, kind, expires_at, used_at
                 FROM messenger_link_codes
                WHERE code = ?""",
            (code.strip().lower(),),
        ).fetchone()
        if not row:
            return None
        if row["used_at"]:
            return None
        if row["kind"] != kind:
            return None
        if row["expires_at"] < now:
            return None
        c.execute(
            "UPDATE messenger_link_codes SET used_at = ? WHERE code = ?",
            (now, code.strip().lower()),
        )
        return int(row["user_id"])


# =========================================================================
#  Links (chat ↔ user)
# =========================================================================

def link_chat(kind: str, chat_id: str, user_id: int, username: str | None = None) -> None:
    """Idempotent: re-linking the same chat to the same user is a no-op,
    re-linking to a different user replaces the old binding."""
    _ensure_tables()
    with connect() as c:
        c.execute(
            """INSERT INTO messenger_links (kind, chat_id, user_id, username)
                   VALUES (?,?,?,?)
               ON CONFLICT(kind, chat_id) DO UPDATE SET
                   user_id = excluded.user_id,
                   username = excluded.username,
                   linked_at = datetime('now')""",
            (kind, str(chat_id), user_id, username),
        )


def unlink_chat(kind: str, chat_id: str) -> None:
    _ensure_tables()
    with connect() as c:
        c.execute("DELETE FROM messenger_links WHERE kind = ? AND chat_id = ?", (kind, str(chat_id)))
        c.execute("DELETE FROM messenger_state WHERE kind = ? AND chat_id = ?", (kind, str(chat_id)))


def unlink_user(user_id: int, kind: str | None = None) -> int:
    """Web-side unlink: remove all (kind, chat_id) bindings for this user.
    Returns the number of links removed."""
    _ensure_tables()
    with connect() as c:
        if kind:
            res = c.execute(
                "DELETE FROM messenger_links WHERE user_id = ? AND kind = ?",
                (user_id, kind),
            )
        else:
            res = c.execute("DELETE FROM messenger_links WHERE user_id = ?", (user_id,))
        return res.rowcount or 0


def resolve_user(kind: str, chat_id: str) -> int | None:
    _ensure_tables()
    with connect() as c:
        row = c.execute(
            "SELECT user_id FROM messenger_links WHERE kind = ? AND chat_id = ?",
            (kind, str(chat_id)),
        ).fetchone()
    return int(row["user_id"]) if row else None


def list_links_for_user(user_id: int) -> list[dict]:
    _ensure_tables()
    with connect() as c:
        rows = c.execute(
            "SELECT kind, chat_id, username, linked_at FROM messenger_links WHERE user_id = ?",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# =========================================================================
#  State machine
# =========================================================================

def get_state(kind: str, chat_id: str) -> dict:
    _ensure_tables()
    cutoff = (datetime.utcnow() - timedelta(minutes=STATE_TTL_MIN)).isoformat()
    with connect() as c:
        row = c.execute(
            """SELECT mode, payload, updated_at FROM messenger_state
                WHERE kind = ? AND chat_id = ?""",
            (kind, str(chat_id)),
        ).fetchone()
    if not row:
        return {"mode": "chat", "payload": None}
    if (row["updated_at"] or "") < cutoff:
        # Expire stale modes back to default chat.
        return {"mode": "chat", "payload": None}
    return {"mode": row["mode"], "payload": row["payload"]}


def set_state(kind: str, chat_id: str, mode: str, payload: str | None = None) -> None:
    _ensure_tables()
    with connect() as c:
        c.execute(
            """INSERT INTO messenger_state (kind, chat_id, mode, payload, updated_at)
                   VALUES (?,?,?,?, datetime('now'))
               ON CONFLICT(kind, chat_id) DO UPDATE SET
                   mode = excluded.mode,
                   payload = excluded.payload,
                   updated_at = datetime('now')""",
            (kind, str(chat_id), mode, payload),
        )


def reset_state(kind: str, chat_id: str) -> None:
    set_state(kind, str(chat_id), "chat", None)


# =========================================================================
#  Socratic preference (persisted across sessions like the web cookie does)
# =========================================================================

def get_pref(kind: str, chat_id: str, key: str, default: str | None = None) -> str | None:
    _ensure_tables()
    with connect() as c:
        row = c.execute(
            "SELECT value FROM messenger_prefs WHERE kind=? AND chat_id=? AND key=?",
            (kind, str(chat_id), key),
        ).fetchone()
    return row["value"] if row else default


def set_pref(kind: str, chat_id: str, key: str, value: str | None) -> None:
    _ensure_tables()
    with connect() as c:
        c.execute(
            """INSERT INTO messenger_prefs (kind, chat_id, key, value, updated_at)
                   VALUES (?, ?, ?, ?, datetime('now'))
               ON CONFLICT(kind, chat_id, key) DO UPDATE SET
                   value = excluded.value,
                   updated_at = datetime('now')""",
            (kind, str(chat_id), key, value),
        )


def get_socratic(kind: str, chat_id: str) -> bool:
    return get_pref(kind, chat_id, "socratic", "off") == "on"


def set_socratic(kind: str, chat_id: str, on: bool) -> None:
    set_pref(kind, chat_id, "socratic", "on" if on else "off")
