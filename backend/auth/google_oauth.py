"""Google OAuth 2.0 authorization-code flow.

Minimal, dependency-free (just httpx). Builds the authorize URL, handles the
callback, exchanges the code for tokens, decodes the ID token, and returns
(google_sub, email, name) for `service.upsert_google_user`.
"""
from __future__ import annotations

import base64
import json
from urllib.parse import urlencode

import httpx

from config import (
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_REDIRECT_URI,
)

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"


def is_configured() -> bool:
    """True iff Google client id + secret are present in the environment."""
    return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)


def authorize_url(state: str) -> str:
    """Build the Google authorize URL with the given anti-CSRF state."""
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    return f"{AUTH_URL}?{urlencode(params)}"


def _decode_id_token(id_token: str) -> dict:
    """Decode the unsigned middle segment of an ID token.

    We trust the token here because we received it directly from the Google
    token endpoint over TLS (the OAuth code exchange). Independent JWKS
    verification would harden this but isn't strictly required for the
    server-side code flow.
    """
    parts = id_token.split(".")
    if len(parts) != 3:
        raise ValueError("Malformed id_token")
    pad = "=" * (-len(parts[1]) % 4)
    payload = base64.urlsafe_b64decode(parts[1] + pad)
    return json.loads(payload)


def exchange_code(code: str) -> tuple[str, str, str | None]:
    """Exchange an authorization code for tokens; return (sub, email, name)."""
    if not is_configured():
        raise RuntimeError("Google OAuth is not configured on this server.")
    with httpx.Client(timeout=15) as c:
        r = c.post(
            TOKEN_URL,
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
        )
        r.raise_for_status()
        tokens = r.json()
    claims = _decode_id_token(tokens["id_token"])
    sub = claims.get("sub")
    email = claims.get("email")
    name = claims.get("name")
    if not sub or not email:
        raise ValueError("Google response missing sub/email")
    return sub, email, name
