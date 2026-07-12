"""Password hashing, JWT issuance/verification, cookie helpers."""
from __future__ import annotations

import secrets
import time
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from config import (
    COOKIE_NAME,
    COOKIE_SECURE,
    JWT_ALGO,
    JWT_EXP_DAYS,
    JWT_SECRET,
)

_HASHER = PasswordHasher()


def hash_password(plain: str) -> str:
    """Argon2id hash of a plaintext password."""
    return _HASHER.hash(plain)


def verify_password(plain: str, hashed: str | None) -> bool:
    """Constant-time verify; False on any mismatch or null hash."""
    if not hashed:
        return False
    try:
        return _HASHER.verify(hashed, plain)
    except VerifyMismatchError:
        return False
    except Exception:
        return False


def _require_secret() -> str:
    """Refuse to issue tokens if JWT_SECRET wasn't set in the environment."""
    if not JWT_SECRET:
        raise RuntimeError(
            "JWT_SECRET is not set. Generate one with `python -c \"import secrets; print(secrets.token_urlsafe(64))\"` "
            "and export it before starting the server."
        )
    return JWT_SECRET


def issue_session(user_id: int, email: str) -> str:
    """Mint a signed JWT for the given user, valid for JWT_EXP_DAYS days."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=JWT_EXP_DAYS)).timestamp()),
    }
    return jwt.encode(payload, _require_secret(), algorithm=JWT_ALGO)


def decode_session(token: str) -> dict | None:
    """Return the JWT payload if valid, else None (any error → None)."""
    if not token:
        return None
    try:
        return jwt.decode(token, _require_secret(), algorithms=[JWT_ALGO])
    except Exception:
        return None


def _cookie_domain_and_secure() -> dict:
    """domain + secure flags shared by set and delete, so both always match."""
    from config import COOKIE_DOMAIN
    kw: dict = {}
    if COOKIE_DOMAIN:
        kw["domain"] = COOKIE_DOMAIN
    if COOKIE_SECURE:
        kw["secure"] = True
    return kw


def cookie_delete_kwargs() -> dict:
    """kwargs for response.delete_cookie — MUST include every attribute
    that was used when setting the cookie (especially domain), otherwise
    the browser won't match and the cookie survives.  The root cause of
    sign-out not working was that logout called delete_cookie(path="/")
    without the domain=.vaaani.in that had been set on login."""
    from config import COOKIE_SAMESITE
    kw = {
        "path": "/",
        "httponly": True,
        "samesite": COOKIE_SAMESITE if COOKIE_SAMESITE in ("lax", "strict", "none") else "lax",
    }
    kw.update(_cookie_domain_and_secure())
    return kw


def cookie_settings() -> dict:
    """Common kwargs for FastAPI's `response.set_cookie`.

    In production, COOKIE_DOMAIN=.vaaani.in shares the session across the Vercel
    frontend (app.vaaani.in) and the GCP backend (api.vaaani.in) — same-site, so
    SameSite=Lax still works and there is no cross-site-cookie problem.
    """
    from config import COOKIE_DOMAIN, COOKIE_SAMESITE
    kw = {
        "key": COOKIE_NAME,
        "httponly": True,
        "secure": COOKIE_SECURE,
        "samesite": COOKIE_SAMESITE if COOKIE_SAMESITE in ("lax", "strict", "none") else "lax",
        "max_age": JWT_EXP_DAYS * 86400,
        "path": "/",
    }
    if COOKIE_DOMAIN:
        kw["domain"] = COOKIE_DOMAIN
    return kw


def random_token(nbytes: int = 32) -> str:
    """URL-safe random token for email verification + OAuth state."""
    return secrets.token_urlsafe(nbytes)


def random_otp(length: int) -> str:
    """Numeric OTP of `length` digits, zero-padded."""
    n = secrets.randbelow(10 ** length)
    return str(n).zfill(length)
