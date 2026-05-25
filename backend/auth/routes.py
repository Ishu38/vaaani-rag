"""FastAPI router for /auth/* endpoints.

Cookie-based JWT auth (httpOnly, SameSite=Lax). All sensitive mutations are
POST. Most endpoints return JSON; `verify-email` returns an HTML redirect to
the front-end verify page so we can show a friendly confirmation.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Cookie, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr, Field

from config import APP_BASE_URL, COOKIE_NAME

from . import github_oauth, google_oauth, school, service
from .db import init_db
from .security import (
    cookie_settings,
    decode_session,
    issue_session,
    random_token,
)

router = APIRouter(prefix="/auth", tags=["auth"])
init_db()


# ---------------- schemas ----------------

class SignupBody(BaseModel):
    """POST /auth/signup body."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=200)
    name: str | None = None


class LoginBody(BaseModel):
    """POST /auth/login body."""
    email: EmailStr
    password: str


class PhoneSendBody(BaseModel):
    """POST /auth/phone/send-otp body."""
    phone: str = Field(..., min_length=6, max_length=20)


class PhoneVerifyBody(BaseModel):
    """POST /auth/phone/verify-otp body."""
    code: str = Field(..., min_length=4, max_length=10)


# ---------------- helpers ----------------

def _set_session(response: Response, user_id: int, email: str) -> None:
    """Mint a JWT for the user and attach it as the session cookie."""
    token = issue_session(user_id, email)
    response.set_cookie(value=token, **cookie_settings())


def _current_user(session_cookie: str | None) -> dict | None:
    """Decode the session cookie and load the user row, or None."""
    payload = decode_session(session_cookie or "")
    if not payload:
        return None
    try:
        return service.get_user_by_id(int(payload["sub"]))
    except (KeyError, ValueError):
        return None


def _require_user(session_cookie: str | None) -> dict:
    """Same as `_current_user` but raises 401 if unauthenticated."""
    user = _current_user(session_cookie)
    if not user:
        raise HTTPException(status_code=401, detail="Not signed in.")
    return user


# ---------------- routes ----------------

@router.post("/signup")
def signup(body: SignupBody, response: Response):
    """Create a password-based account and dispatch a verification email."""
    try:
        user = service.create_user_with_password(body.email, body.password, body.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _set_session(response, user["id"], user["email"])
    return {"user": user, "next": "verify-email-pending"}


@router.post("/login")
def login(body: LoginBody, response: Response):
    """Authenticate with email + password; sets the session cookie."""
    user = service.authenticate_password(body.email, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    _set_session(response, user["id"], user["email"])
    return {"user": user}


@router.post("/logout")
def logout(response: Response):
    """Clear the session cookie."""
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/me")
def me(vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME)):
    """Return the currently signed-in user, or {user: null}."""
    return {"user": _current_user(vaaani_session)}


@router.post("/resend-verification")
def resend_verification(vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME)):
    """Send the verification email again (only for unverified users)."""
    user = _require_user(vaaani_session)
    if user["email_verified_at"]:
        return {"ok": True, "already_verified": True}
    service.send_email_verification(user["id"], user["email"])
    return {"ok": True}


@router.get("/verify-email/{token}")
def verify_email(token: str):
    """Consume an email verification token; redirect to a friendly status page."""
    user = service.verify_email_token(token)
    target = "/verify?status=ok" if user else "/verify?status=invalid"
    return RedirectResponse(url=target, status_code=303)


@router.post("/phone/send-otp")
def phone_send(
    body: PhoneSendBody,
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """Issue an OTP to the user's phone (real SMS only if provider configured)."""
    user = _require_user(vaaani_session)
    service.send_phone_otp(user["id"], body.phone)
    return {"ok": True}


@router.post("/phone/verify-otp")
def phone_verify(
    body: PhoneVerifyBody,
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """Confirm an OTP and mark the user's phone verified."""
    user = _require_user(vaaani_session)
    ok = service.verify_phone_otp(user["id"], body.code)
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid or expired code.")
    return {"ok": True, "user": service.get_user_by_id(user["id"])}


# ---------------- Google OAuth ----------------

@router.get("/google/configured")
def google_configured():
    """Public probe so the front-end knows whether to render the Google button."""
    return {"configured": google_oauth.is_configured()}


@router.get("/google/start")
def google_start(request: Request):
    """Begin the Google authorize flow; stash anti-CSRF state in a cookie."""
    if not google_oauth.is_configured():
        raise HTTPException(status_code=503, detail="Google sign-in is not configured on this server.")
    state = random_token(24)
    resp = RedirectResponse(url=google_oauth.authorize_url(state), status_code=303)
    resp.set_cookie("oauth_state", state, httponly=True, samesite="lax", max_age=600, path="/")
    return resp


@router.get("/google/callback")
def google_callback(request: Request, code: str = "", state: str = ""):
    """Handle Google's redirect; exchange code → upsert user → set session."""
    cookie_state = request.cookies.get("oauth_state")
    if not code or not state or state != cookie_state:
        raise HTTPException(status_code=400, detail="Invalid OAuth state.")
    try:
        sub, email, name = google_oauth.exchange_code(code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Google sign-in failed: {e}")
    user = service.upsert_google_user(sub, email, name)
    resp = RedirectResponse(url="/account?welcome=1", status_code=303)
    resp.delete_cookie("oauth_state", path="/")
    token = issue_session(user["id"], user["email"])
    resp.set_cookie(value=token, **cookie_settings())
    return resp


# ---------------- GitHub OAuth ----------------

@router.get("/github/configured")
def github_configured():
    """Public probe so the front-end knows whether to render the GitHub button."""
    return {"configured": github_oauth.is_configured()}


@router.get("/github/start")
def github_start(request: Request):
    """Begin the GitHub authorize flow; stash anti-CSRF state in a cookie."""
    if not github_oauth.is_configured():
        raise HTTPException(status_code=503, detail="GitHub sign-in is not configured on this server.")
    state = random_token(24)
    resp = RedirectResponse(url=github_oauth.authorize_url(state), status_code=303)
    resp.set_cookie("oauth_state", state, httponly=True, samesite="lax", max_age=600, path="/")
    return resp


@router.get("/github/callback")
def github_callback(request: Request, code: str = "", state: str = ""):
    """Handle GitHub's redirect; exchange code → upsert user → set session."""
    cookie_state = request.cookies.get("oauth_state")
    if not code or not state or state != cookie_state:
        raise HTTPException(status_code=400, detail="Invalid OAuth state.")
    try:
        gh_id, email, name = github_oauth.exchange_code(code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"GitHub sign-in failed: {e}")
    user = service.upsert_github_user(gh_id, email, name)
    resp = RedirectResponse(url="/account?welcome=1", status_code=303)
    resp.delete_cookie("oauth_state", path="/")
    token = issue_session(user["id"], user["email"])
    resp.set_cookie(value=token, **cookie_settings())
    return resp


# ---------------- school org routes ----------------


class CreateSchoolBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    plan: str = "school_trial"


class JoinSchoolBody(BaseModel):
    code: str = Field(..., min_length=4, max_length=20)


class GuardrailsBody(BaseModel):
    curriculum: str | None = None
    socratic_level: str | None = None
    allow_direct_answers: bool | None = None
    allowed_subjects: list[str] | None = None
    grade_level: str | None = None
    board: str | None = None


@router.get("/schools")
def list_my_schools(vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME)):
    """Return all schools the current user belongs to."""
    user = _require_user(vaaani_session)
    return {"schools": school.list_schools_for_user(user["id"])}


@router.post("/schools")
def create_school(
    body: CreateSchoolBody,
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """Create a new school org. The creator becomes the admin."""
    user = _require_user(vaaani_session)
    try:
        s = school.create_school(body.name, user["id"], body.plan)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"school": s}


@router.get("/schools/{school_id}")
def get_school_detail(
    school_id: int,
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """Get school details (must be a member)."""
    user = _require_user(vaaani_session)
    role = school.get_user_role(user["id"], school_id)
    if not role:
        raise HTTPException(status_code=403, detail="Not a member of this school.")
    s = school.get_school(school_id)
    if not s:
        raise HTTPException(status_code=404, detail="School not found.")
    return {"school": s, "role": role}


@router.post("/schools/join")
def join_school(
    body: JoinSchoolBody,
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """Join a school by invite code."""
    user = _require_user(vaaani_session)
    result = school.join_school(user["id"], body.code)
    if not result:
        raise HTTPException(status_code=404, detail="Invalid school code.")
    return {"school": result}


@router.get("/schools/{school_id}/members")
def get_members(
    school_id: int,
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """List all members of a school (staff only)."""
    user = _require_user(vaaani_session)
    if not school.is_school_staff(user["id"], school_id):
        raise HTTPException(status_code=403, detail="Only teachers and admins can view members.")
    return {
        "members": school.list_members(school_id),
        "counts": school.count_members_by_role(school_id),
    }


@router.delete("/schools/{school_id}/members/{member_id}")
def remove_member(
    school_id: int,
    member_id: int,
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """Remove a member from the school (admin only)."""
    user = _require_user(vaaani_session)
    if not school.is_school_admin(user["id"], school_id):
        raise HTTPException(status_code=403, detail="Only admins can remove members.")
    if member_id == user["id"]:
        raise HTTPException(status_code=400, detail="You cannot remove yourself. Transfer admin first.")
    if not school.remove_member(school_id, member_id):
        raise HTTPException(status_code=404, detail="Member not found.")
    return {"ok": True}


@router.get("/schools/{school_id}/guardrails")
def get_guardrails(
    school_id: int,
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """View a school's guardrail settings (staff only)."""
    user = _require_user(vaaani_session)
    if not school.is_school_staff(user["id"], school_id):
        raise HTTPException(status_code=403, detail="Only teachers and admins can view guardrails.")
    return {"guardrails": school.get_guardrails(school_id)}


@router.put("/schools/{school_id}/guardrails")
def update_guardrails(
    school_id: int,
    body: GuardrailsBody,
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """Update a school's guardrail settings (admin only)."""
    user = _require_user(vaaani_session)
    if not school.is_school_admin(user["id"], school_id):
        raise HTTPException(status_code=403, detail="Only admins can configure guardrails.")
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        g = school.set_guardrails(school_id, updates)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"guardrails": g}


@router.get("/schools/{school_id}/dashboard")
def get_school_dashboard(
    school_id: int,
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """Aggregated dashboard data for a school (staff only)."""
    user = _require_user(vaaani_session)
    if not school.is_school_staff(user["id"], school_id):
        raise HTTPException(status_code=403, detail="Only teachers and admins can view the dashboard.")
    return school.school_dashboard(school_id)
