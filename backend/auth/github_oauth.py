"""GitHub OAuth 2.0 authorization-code flow.

Mirrors auth/google_oauth.py. GitHub doesn't return an ID token, so after the
token exchange we hit the REST API to fetch the user profile + verified email.
"""
from __future__ import annotations

from urllib.parse import urlencode

import httpx

from config import (
    GITHUB_CLIENT_ID,
    GITHUB_CLIENT_SECRET,
    GITHUB_REDIRECT_URI,
)

AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
TOKEN_URL = "https://github.com/login/oauth/access_token"
API_USER_URL = "https://api.github.com/user"
API_EMAILS_URL = "https://api.github.com/user/emails"


def is_configured() -> bool:
    """True iff GitHub client id + secret are present in the environment."""
    return bool(GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET)


def authorize_url(state: str) -> str:
    """Build the GitHub authorize URL with the given anti-CSRF state."""
    params = {
        "client_id": GITHUB_CLIENT_ID,
        "redirect_uri": GITHUB_REDIRECT_URI,
        "scope": "read:user user:email",
        "state": state,
        "allow_signup": "true",
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def _primary_verified_email(emails: list[dict]) -> str | None:
    """Pick the user's primary+verified email; fall back to any verified one."""
    for e in emails:
        if e.get("primary") and e.get("verified") and e.get("email"):
            return e["email"]
    for e in emails:
        if e.get("verified") and e.get("email"):
            return e["email"]
    return None


def exchange_code(code: str) -> tuple[str, str, str | None]:
    """Exchange an authorization code for an access token; return (github_id, email, name).

    `github_id` is GitHub's stable numeric user id, stringified — safe to put
    in the users.github_id column even if a user later renames their handle.
    """
    if not is_configured():
        raise RuntimeError("GitHub OAuth is not configured on this server.")
    with httpx.Client(timeout=15) as c:
        # 1. Exchange code → access_token. GitHub returns JSON when we set the
        #    Accept header explicitly (default is form-urlencoded).
        r = c.post(
            TOKEN_URL,
            data={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": GITHUB_REDIRECT_URI,
            },
            headers={"Accept": "application/json"},
        )
        r.raise_for_status()
        token = r.json().get("access_token")
        if not token:
            raise ValueError(f"GitHub token endpoint returned no access_token: {r.text[:200]}")

        # 2. Fetch user profile.
        auth = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
        u = c.get(API_USER_URL, headers=auth)
        u.raise_for_status()
        profile = u.json()
        gh_id = profile.get("id")
        name = profile.get("name") or profile.get("login")
        email = profile.get("email")  # may be None if user kept it private

        # 3. If profile email is hidden, look it up via /user/emails (needs user:email scope).
        if not email:
            e = c.get(API_EMAILS_URL, headers=auth)
            e.raise_for_status()
            email = _primary_verified_email(e.json() or [])

    if not gh_id or not email:
        raise ValueError("GitHub response missing id or verified email")
    return str(gh_id), email, name
