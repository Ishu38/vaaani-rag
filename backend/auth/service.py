"""Business logic for accounts, verification, and OTP.

The router calls these functions; they encapsulate every DB read/write so
the routes stay thin and testable.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from config import (
    APP_BASE_URL,
    DEFAULT_PLAN,
    EMAIL_VERIFY_EXP_HOURS,
    PHONE_OTP_EXP_MIN,
    PHONE_OTP_LEN,
)

from .db import connect
from .email_sender import get_sender as get_email_sender
from .security import hash_password, random_otp, random_token, verify_password
from .sms_sender import get_sender as get_sms_sender


# ---------------- helpers ----------------

def _now_iso() -> str:
    """UTC ISO timestamp without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _hash_otp(code: str) -> str:
    """SHA-256 the OTP so leaked DB rows don't reveal recent codes."""
    return hashlib.sha256(code.encode()).hexdigest()


def _row_to_user(row) -> dict:
    """Project a sqlite3.Row into a JSON-friendly user dict (sans secrets)."""
    keys = row.keys()
    return {
        "id": row["id"],
        "email": row["email"],
        "name": row["name"],
        "phone": row["phone"],
        "plan": row["plan"],
        "email_verified_at": row["email_verified_at"],
        "phone_verified_at": row["phone_verified_at"],
        "google_linked": bool(row["google_sub"]),
        "github_linked": bool(row["github_id"]) if "github_id" in keys else False,
        "created_at": row["created_at"],
        # DPDP fields — read by auth/dpdp.allow_processing() and the SPA's
        # consent modal. All optional so older rows (pre-2026-05-26) still
        # round-trip cleanly with consent_status defaulting to 'not_required'.
        "date_of_birth": row["date_of_birth"] if "date_of_birth" in keys else None,
        "consent_status": (row["consent_status"] if "consent_status" in keys else None) or "not_required",
        "active_consent_id": row["active_consent_id"] if "active_consent_id" in keys else None,
        "deleted_at": row["deleted_at"] if "deleted_at" in keys else None,
    }


# ---------------- user lookup ----------------

def get_user_by_id(uid: int) -> dict | None:
    """Fetch a user by primary key, returning the projected dict or None."""
    with connect() as c:
        r = c.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
        return _row_to_user(r) if r else None


def get_user_by_email(email: str) -> dict | None:
    """Fetch a user by email (case-insensitive) or None."""
    with connect() as c:
        r = c.execute("SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email.strip(),)).fetchone()
        return _row_to_user(r) if r else None


def _get_raw_by_email(email: str):
    """Internal: get the raw row (with password_hash) for password verification."""
    with connect() as c:
        return c.execute("SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email.strip(),)).fetchone()


# ---------------- signup ----------------

def create_user_with_password(
    email: str,
    password: str,
    name: str | None = None,
    date_of_birth: str | None = None,
) -> dict:
    """Insert a new password-based user. Raises ValueError if email taken.

    `date_of_birth` is ISO YYYY-MM-DD. Under-18s are marked consent_status =
    'pending' so the DPDP gate refuses /chat /ingest until a parent grants
    consent via the magic-link flow (see auth/dpdp.py).
    """
    from .dpdp import is_minor  # local import — avoid module-load cycle
    email = email.strip().lower()
    name = (name or "").strip() or None
    dob = (date_of_birth or "").strip() or None
    if get_user_by_email(email):
        raise ValueError("An account with that email already exists.")
    pw = hash_password(password)
    consent_status = "pending" if is_minor(dob) else "not_required"
    with connect() as c:
        c.execute(
            "INSERT INTO users (email, password_hash, name, plan, date_of_birth, consent_status) "
            "VALUES (?,?,?,?,?,?)",
            (email, pw, name, DEFAULT_PLAN, dob, consent_status),
        )
    user = get_user_by_email(email)
    assert user is not None
    send_email_verification(user["id"], email)
    return user


def authenticate_password(email: str, password: str) -> dict | None:
    """Return the projected user if password matches, else None."""
    row = _get_raw_by_email(email)
    if not row or not verify_password(password, row["password_hash"]):
        return None
    with connect() as c:
        c.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (_now_iso(), row["id"]))
    return _row_to_user(row)


# ---------------- email verification ----------------

def send_email_verification(user_id: int, email: str) -> None:
    """Generate a verification token, persist it, and dispatch the email."""
    token = random_token()
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=EMAIL_VERIFY_EXP_HOURS)).isoformat()
    with connect() as c:
        c.execute(
            "INSERT INTO email_verification_tokens (token, user_id, expires_at) VALUES (?,?,?)",
            (token, user_id, expires_at),
        )
    link = f"{APP_BASE_URL.rstrip('/')}/verify?token={token}"
    body_text = (
        f"Welcome to Vaaani.\n\n"
        f"Click the link below to verify your email address:\n{link}\n\n"
        f"This link expires in {EMAIL_VERIFY_EXP_HOURS} hours. "
        f"If you didn't sign up, ignore this email."
    )
    body_html = (
        f"<p>Welcome to <strong>Vaaani</strong>.</p>"
        f"<p>Click the link below to verify your email address:</p>"
        f"<p><a href='{link}'>{link}</a></p>"
        f"<p>This link expires in {EMAIL_VERIFY_EXP_HOURS} hours.</p>"
    )
    get_email_sender().send(email, "Verify your Vaaani account", body_text, body_html)


PASSWORD_RESET_EXP_HOURS = 1


def send_password_reset(email: str) -> bool:
    """Issue a password-reset token for `email` and dispatch the reset email.

    Returns True if an account existed (and mail was dispatched), False otherwise.
    Callers MUST NOT leak this back to the client verbatim — always show the same
    generic response to avoid revealing which emails have accounts.
    """
    user = get_user_by_email((email or "").strip())
    if not user:
        return False
    token = random_token()
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=PASSWORD_RESET_EXP_HOURS)).isoformat()
    with connect() as c:
        c.execute(
            "INSERT INTO password_reset_tokens (token, user_id, expires_at) VALUES (?,?,?)",
            (token, user["id"], expires_at),
        )
    link = f"{APP_BASE_URL.rstrip('/')}/reset-password?token={token}"
    body_text = (
        f"We received a request to reset your Vaaani password.\n\n"
        f"Click the link below to choose a new password:\n{link}\n\n"
        f"This link expires in {PASSWORD_RESET_EXP_HOURS} hour(s). "
        f"If you didn't request this, ignore this email — your password stays unchanged."
    )
    body_html = (
        f"<p>We received a request to reset your <strong>Vaaani</strong> password.</p>"
        f"<p>Click the link below to choose a new password:</p>"
        f"<p><a href='{link}'>{link}</a></p>"
        f"<p>This link expires in {PASSWORD_RESET_EXP_HOURS} hour(s). "
        f"If you didn't request this, you can safely ignore this email.</p>"
    )
    get_email_sender().send(email, "Reset your Vaaani password", body_text, body_html)
    return True


def reset_password(token: str, new_password: str) -> dict | None:
    """Consume a reset token and set a new password. Returns the user or None."""
    if not token:
        return None
    with connect() as c:
        row = c.execute(
            "SELECT * FROM password_reset_tokens WHERE token = ?", (token,)
        ).fetchone()
        if not row or row["used_at"] is not None:
            return None
        if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
            return None
        c.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (hash_password(new_password), row["user_id"]),
        )
        c.execute("UPDATE password_reset_tokens SET used_at = ? WHERE token = ?", (_now_iso(), token))
        # Invalidate any other outstanding reset tokens for this user.
        c.execute(
            "UPDATE password_reset_tokens SET used_at = ? WHERE user_id = ? AND used_at IS NULL",
            (_now_iso(), row["user_id"]),
        )
    return get_user_by_id(row["user_id"])


def verify_email_token(token: str) -> dict | None:
    """Consume an email verification token; mark the user verified. Returns user dict or None."""
    if not token:
        return None
    with connect() as c:
        row = c.execute(
            "SELECT * FROM email_verification_tokens WHERE token = ?", (token,)
        ).fetchone()
        if not row or row["used_at"] is not None:
            return None
        if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
            return None
        c.execute("UPDATE email_verification_tokens SET used_at = ? WHERE token = ?", (_now_iso(), token))
        c.execute(
            "UPDATE users SET email_verified_at = COALESCE(email_verified_at, ?) WHERE id = ?",
            (_now_iso(), row["user_id"]),
        )
    return get_user_by_id(row["user_id"])


# ---------------- phone OTP ----------------

def send_phone_otp(user_id: int, phone: str) -> None:
    """Generate a 6-digit OTP, persist its hash, dispatch via the SMS sender."""
    phone = phone.strip()
    code = random_otp(PHONE_OTP_LEN)
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=PHONE_OTP_EXP_MIN)).isoformat()
    with connect() as c:
        # Invalidate any pending OTPs for this user/phone first.
        c.execute(
            "UPDATE phone_otps SET used_at = ? WHERE user_id = ? AND phone = ? AND used_at IS NULL",
            (_now_iso(), user_id, phone),
        )
        c.execute(
            "INSERT INTO phone_otps (user_id, phone, code_hash, expires_at) VALUES (?,?,?,?)",
            (user_id, phone, _hash_otp(code), expires_at),
        )
        c.execute("UPDATE users SET phone = ? WHERE id = ?", (phone, user_id))
    get_sms_sender().send_otp(phone, code)


def verify_phone_otp(user_id: int, code: str) -> bool:
    """Match the latest unused OTP for `user_id`. Marks phone verified on success."""
    if not code or not code.strip().isdigit():
        return False
    code = code.strip()
    with connect() as c:
        row = c.execute(
            "SELECT * FROM phone_otps WHERE user_id = ? AND used_at IS NULL "
            "ORDER BY id DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        if not row:
            return False
        if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
            return False
        if row["attempts"] >= 5:
            return False
        if _hash_otp(code) != row["code_hash"]:
            c.execute("UPDATE phone_otps SET attempts = attempts + 1 WHERE id = ?", (row["id"],))
            return False
        c.execute("UPDATE phone_otps SET used_at = ? WHERE id = ?", (_now_iso(), row["id"]))
        c.execute(
            "UPDATE users SET phone_verified_at = COALESCE(phone_verified_at, ?) WHERE id = ?",
            (_now_iso(), user_id),
        )
    return True


# ---------------- google oauth ----------------

def upsert_google_user(google_sub: str, email: str, name: str | None) -> dict:
    """Create or link a Google account; Google emails are verified by Google."""
    email = email.strip().lower()
    with connect() as c:
        existing = c.execute(
            "SELECT id FROM users WHERE google_sub = ? OR email = ? COLLATE NOCASE LIMIT 1",
            (google_sub, email),
        ).fetchone()
        if existing:
            c.execute(
                "UPDATE users SET google_sub = COALESCE(google_sub, ?), "
                "email_verified_at = COALESCE(email_verified_at, ?), "
                "name = COALESCE(name, ?), last_login_at = ? WHERE id = ?",
                (google_sub, _now_iso(), name, _now_iso(), existing["id"]),
            )
            uid = existing["id"]
        else:
            c.execute(
                "INSERT INTO users (email, name, google_sub, email_verified_at, plan, last_login_at) "
                "VALUES (?,?,?,?,?,?)",
                (email, name, google_sub, _now_iso(), DEFAULT_PLAN, _now_iso()),
            )
            uid = c.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    user = get_user_by_id(uid)
    assert user is not None
    return user


def upsert_github_user(github_id: str, email: str, name: str | None) -> dict:
    """Create or link a GitHub account; we trust the verified email from the GitHub API."""
    email = email.strip().lower()
    with connect() as c:
        existing = c.execute(
            "SELECT id FROM users WHERE github_id = ? OR email = ? COLLATE NOCASE LIMIT 1",
            (github_id, email),
        ).fetchone()
        if existing:
            c.execute(
                "UPDATE users SET github_id = COALESCE(github_id, ?), "
                "email_verified_at = COALESCE(email_verified_at, ?), "
                "name = COALESCE(name, ?), last_login_at = ? WHERE id = ?",
                (github_id, _now_iso(), name, _now_iso(), existing["id"]),
            )
            uid = existing["id"]
        else:
            c.execute(
                "INSERT INTO users (email, name, github_id, email_verified_at, plan, last_login_at) "
                "VALUES (?,?,?,?,?,?)",
                (email, name, github_id, _now_iso(), DEFAULT_PLAN, _now_iso()),
            )
            uid = c.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    user = get_user_by_id(uid)
    assert user is not None
    return user
