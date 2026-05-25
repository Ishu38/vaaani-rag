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


def cookie_settings() -> dict:
    """Common kwargs for FastAPI's `response.set_cookie`."""
    return {
        "key": COOKIE_NAME,
        "httponly": True,
        "secure": COOKIE_SECURE,
        "samesite": "lax",
        "max_age": JWT_EXP_DAYS * 86400,
        "path": "/",
    }


def random_token(nbytes: int = 32) -> str:
    """URL-safe random token for email verification + OAuth state."""
    return secrets.token_urlsafe(nbytes)


def random_otp(length: int) -> str:
    """Numeric OTP of `length` digits, zero-padded."""
    n = secrets.randbelow(10 ** length)
    return str(n).zfill(length)
