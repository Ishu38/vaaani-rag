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
    cookie_delete_kwargs,
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
    # ISO YYYY-MM-DD. Required for DPDP §9 age determination — the signup
    # route 400s if missing. Caller can still create OAuth accounts without
    # one, in which case the user falls into the safer 'is_minor=True' bucket
    # until they fill DOB on the account page.
    date_of_birth: str | None = None
    # Under-18 signups must include parent contact so we can dispatch the
    # consent magic-link immediately. Adults can omit these.
    parent_email: EmailStr | None = None
    parent_name: str | None = None
    parent_phone: str | None = None


class LoginBody(BaseModel):
    """POST /auth/login body."""
    email: EmailStr
    password: str


class ForgotPasswordBody(BaseModel):
    """POST /auth/forgot-password body."""
    email: EmailStr


class ResetPasswordBody(BaseModel):
    """POST /auth/reset-password body."""
    token: str = Field(..., min_length=8, max_length=200)
    password: str = Field(..., min_length=8, max_length=200)


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
def signup(body: SignupBody, request: Request, response: Response):
    """Create a password-based account and dispatch a verification email.

    DPDP §9 branch: if the supplied DOB makes the user under-18, we mark the
    account consent_status='pending', dispatch a parental consent magic-link
    to the parent_email, and DO NOT issue a session cookie. The child has to
    log in only AFTER the parent confirms. Adults proceed normally.
    """
    from . import dpdp
    dob = (body.date_of_birth or "").strip()
    # DOB is mandatory on the signup form — DPDP §9 needs an age to apply.
    # We accept a missing DOB only for legacy/oauth flows handled elsewhere.
    if not dob:
        raise HTTPException(status_code=400, detail="Date of birth is required (DPDP §9 age verification).")
    age = dpdp.age_from_dob(dob)
    if age is None:
        raise HTTPException(status_code=400, detail="Date of birth must be in YYYY-MM-DD format.")
    if age > 130 or age < 0:
        raise HTTPException(status_code=400, detail="Date of birth is out of range.")
    minor = age < 18
    if minor and not body.parent_email:
        raise HTTPException(
            status_code=400,
            detail="Under-18 signups must include parent_email so we can request parental consent (DPDP §9).",
        )

    try:
        user = service.create_user_with_password(
            body.email, body.password, body.name, date_of_birth=dob,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if minor:
        # Don't set a session — child can't sign in until parent confirms.
        actor_ip = request.client.host if request.client else None
        try:
            consent = dpdp.request_parental_consent(
                user["id"],
                body.parent_email,
                parent_name=body.parent_name,
                parent_phone=body.parent_phone,
                actor_ip=actor_ip,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {
            "user": user,
            "next": "parental-consent-pending",
            "consent": consent,
            "message": (
                f"Account created. We've emailed {body.parent_email} a verifiable "
                f"consent request — your account stays locked until your parent confirms."
            ),
        }

    _set_session(response, user["id"], user["email"])
    return {"user": user, "next": "verify-email-pending"}


@router.post("/login")
def login(body: LoginBody, response: Response):
    """Authenticate with email + password; sets the session cookie.

    DPDP gates: refuse login for soft-deleted accounts (§12 erasure honoured)
    and for under-18s whose parental consent is still pending or withdrawn.
    The login surfaces an explicit reason so the SPA can route the user
    to the right next step (resend consent / contact support).
    """
    user = service.authenticate_password(body.email, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if user.get("deleted_at"):
        raise HTTPException(status_code=403, detail="This account has been deleted.")
    status = user.get("consent_status") or "not_required"
    if status == "pending":
        raise HTTPException(
            status_code=403,
            detail="parental_consent_pending: your parent must confirm consent before you can sign in. Check the email we sent them.",
        )
    if status == "withdrawn":
        raise HTTPException(
            status_code=403,
            detail="parental_consent_withdrawn: your parent has withdrawn consent. Contact iamanushka32@gmail.com to restore access.",
        )
    _set_session(response, user["id"], user["email"])
    return {"user": user}


@router.post("/forgot-password")
def forgot_password(body: ForgotPasswordBody):
    """Start password recovery: email a reset link if the account exists.

    Always returns the same generic response so the endpoint can't be used to
    discover which emails have accounts (no user enumeration).
    """
    try:
        service.send_password_reset(body.email)
    except Exception:
        pass  # never leak internal state / existence via error shape
    return {
        "ok": True,
        "message": "If an account exists for that email, we've sent a link to reset the password.",
    }


@router.post("/reset-password")
def reset_password(body: ResetPasswordBody, response: Response):
    """Consume a reset token, set the new password, and sign the user in."""
    user = service.reset_password(body.token, body.password)
    if not user:
        raise HTTPException(status_code=400, detail="This reset link is invalid or has expired. Request a new one.")
    if user.get("deleted_at"):
        raise HTTPException(status_code=403, detail="This account has been deleted.")
    status = user.get("consent_status") or "not_required"
    if status in ("pending", "withdrawn"):
        # Password changed, but consent gate still applies — don't issue a session.
        return {"ok": True, "next": "consent-" + status}
    _set_session(response, user["id"], user["email"])
    return {"ok": True, "user": user}


@router.post("/logout")
def logout(response: Response):
    """Clear the session cookie."""
    response.delete_cookie(COOKIE_NAME, **cookie_delete_kwargs())
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
    resp = RedirectResponse(url="/app?welcome=1", status_code=303)
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
    resp = RedirectResponse(url="/app?welcome=1", status_code=303)
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


# ---------------- parent ↔ student linkage ----------------

class ParentLinkBody(BaseModel):
    parent_email: str
    student_email: str


@router.post("/schools/{school_id}/parent-links")
def create_parent_link(
    school_id: int,
    body: ParentLinkBody,
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """Admin creates a parent ↔ student link inside their school.

    Both users must already be members of the school with the correct roles
    (parent has role='parent', student has role='student'). The admin invites
    them separately via the school invite code — this endpoint only wires up
    the linkage after both accept.
    """
    user = _require_user(vaaani_session)
    if not school.is_school_admin(user["id"], school_id):
        raise HTTPException(status_code=403, detail="Only school admins can link parents to students.")
    parent_user = service.get_user_by_email(body.parent_email.strip().lower())
    student_user = service.get_user_by_email(body.student_email.strip().lower())
    if not parent_user:
        raise HTTPException(status_code=404, detail=f"No account found for parent email {body.parent_email}. Ask them to sign up + join the school first.")
    if not student_user:
        raise HTTPException(status_code=404, detail=f"No account found for student email {body.student_email}. Ask them to sign up + join the school first.")
    link = school.link_parent_to_student(parent_user["id"], student_user["id"], school_id)
    if not link:
        raise HTTPException(
            status_code=400,
            detail="Linkage failed — confirm the parent has joined this school as 'parent' and the student has joined as 'student'.",
        )
    return {"link": link}


@router.delete("/schools/{school_id}/parent-links")
def delete_parent_link(
    school_id: int,
    body: ParentLinkBody,
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """Admin removes a parent ↔ student link."""
    user = _require_user(vaaani_session)
    if not school.is_school_admin(user["id"], school_id):
        raise HTTPException(status_code=403, detail="Only school admins can manage parent links.")
    parent_user = service.get_user_by_email(body.parent_email.strip().lower())
    student_user = service.get_user_by_email(body.student_email.strip().lower())
    if not parent_user or not student_user:
        raise HTTPException(status_code=404, detail="One or both accounts not found.")
    ok = school.unlink_parent_from_student(parent_user["id"], student_user["id"], school_id)
    return {"removed": ok}


@router.get("/schools/{school_id}/parent-dashboard")
def get_parent_dashboard(
    school_id: int,
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """Parent's view of their school: their linked children + school-level summary.
    Available to any member of the school whose role is 'parent'."""
    user = _require_user(vaaani_session)
    role = school.get_user_role(user["id"], school_id)
    if role != "parent":
        raise HTTPException(status_code=403, detail="Only parent members can view the parent dashboard.")
    return school.parent_dashboard(user["id"], school_id)


# =====================================================================
#                 DPDP Act 2023 — parental consent + data rights
# =====================================================================

class ConsentRequestBody(BaseModel):
    parent_email: EmailStr
    parent_name: str | None = None
    parent_phone: str | None = None


class ConsentConfirmBody(BaseModel):
    token: str = Field(..., min_length=8, max_length=200)
    parent_name: str | None = None


class ConsentWithdrawBody(BaseModel):
    # Either the consent_id (parent acting from email link) or implicit
    # (child / parent acting via session — withdraws their active consent).
    consent_id: int | None = None


@router.post("/consent/request")
def consent_request(
    body: ConsentRequestBody,
    request: Request,
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """Re-issue a parental consent link for the current under-18 user.

    Useful when the original email bounced or the token expired. Authenticated
    OR allowed for users in 'pending' state (who cannot otherwise sign in).
    """
    from . import dpdp
    # Allow lookup by email if the child can't sign in (consent_status='pending'
    # blocks /login). Caller must pass their own email in the body in that case
    # — but we keep this endpoint session-only for now to avoid open-resend
    # enumeration. Users with bounced emails should contact support.
    user = _require_user(vaaani_session)
    if not dpdp.is_minor(user.get("date_of_birth")):
        raise HTTPException(status_code=400, detail="Account is 18+, no parental consent required.")
    actor_ip = request.client.host if request.client else None
    consent = dpdp.request_parental_consent(
        user["id"], body.parent_email,
        parent_name=body.parent_name, parent_phone=body.parent_phone,
        actor_ip=actor_ip,
    )
    return {"consent": consent, "message": "Parental consent email re-sent."}


@router.get("/consent/lookup/{token}")
def consent_lookup(token: str):
    """Return the consent record for a token, for the parent landing page to
    render. Returns 404 if the token is bogus. The record includes status
    (pending/granted/withdrawn/expired) so the page can branch."""
    from . import dpdp
    row = dpdp.lookup_consent_by_token(token)
    if not row:
        raise HTTPException(status_code=404, detail="Invalid or expired consent link.")
    from .db import connect
    from datetime import datetime as _dt, timezone as _tz
    child = None
    with connect() as c:
        r = c.execute(
            "SELECT name, email, date_of_birth FROM users WHERE id = ?",
            (row["child_user_id"],),
        ).fetchone()
        if r:
            child = {
                "name": r["name"],
                # Show only the email's local-part — masks the full student
                # address if a parent forwards the link.
                "email_hint": (r["email"].split("@", 1)[0] + "@…") if r["email"] else None,
                "date_of_birth": r["date_of_birth"],
            }
    expired = _dt.fromisoformat(row["expires_at"]) < _dt.now(_tz.utc)
    return {
        "child": child,
        "parent_email": row["parent_email"],
        "consent_text": dpdp.CONSENT_TEXT,
        "consent_text_version": row["consent_text_version"],
        "requested_at": row["requested_at"],
        "granted_at": row["granted_at"],
        "withdrawn_at": row["withdrawn_at"],
        "expires_at": row["expires_at"],
        "expired": expired,
        "status": (
            "granted" if row["granted_at"]
            else "withdrawn" if row["withdrawn_at"]
            else "expired" if expired
            else "pending"
        ),
    }


@router.post("/consent/confirm")
def consent_confirm(body: ConsentConfirmBody, request: Request):
    """Parent endpoint — confirms the consent associated with the token.

    Records parent IP + user-agent for audit. After this returns 200, the
    child account flips to consent_status='granted' and can sign in.
    """
    from . import dpdp
    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent", "")
    try:
        consent = dpdp.confirm_consent(body.token, body.parent_name, ip=ip, user_agent=ua)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "consent": consent}


@router.post("/consent/withdraw")
def consent_withdraw(
    body: ConsentWithdrawBody,
    request: Request,
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """Withdraw an active consent.

    Two paths:
      1. Session-authenticated child/parent → withdraws their account's
         active_consent_id. The session is then invalidated by the next
         request (login blocked for status='withdrawn').
      2. Anonymous + consent_id → only allowed if the caller can prove
         possession via a separate withdrawal token (not implemented yet —
         we'd email a one-time withdraw link). For now this path is gated
         by session.
    """
    from . import dpdp
    user = _require_user(vaaani_session)
    consent_id = body.consent_id or user.get("active_consent_id")
    if not consent_id:
        raise HTTPException(status_code=400, detail="No active consent to withdraw.")
    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent", "")
    consent = dpdp.withdraw_consent(consent_id, actor_user_id=user["id"], ip=ip, user_agent=ua)
    return {"ok": True, "consent": consent}


@router.get("/consent/status")
def consent_status(vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME)):
    """Return the consent state for the current user. Used by the SPA to
    decide whether to show the 'awaiting parent' banner or the normal UI."""
    from . import dpdp
    user = _require_user(vaaani_session)
    return dpdp.consent_status_for_user(user)


# ---------------- §11 right to access / §12 right to erase ----------------

@router.get("/data-export")
def data_export(
    request: Request,
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """Return a JSON document containing everything we hold about the user.

    DPDP §11 right to information about personal data. Streamed as a file
    download so the user can keep it. We log every export to the audit log
    so the user has a paper trail of when they exercised this right.
    """
    from . import dpdp
    from fastapi.responses import JSONResponse
    user = _require_user(vaaani_session)
    payload = dpdp.export_user_data(user["id"])
    ip = request.client.host if request.client else None
    dpdp.audit(user["id"], "data_exported", "via /auth/data-export", ip=ip)
    return JSONResponse(
        content=payload,
        headers={
            "Content-Disposition": f"attachment; filename=\"vaaani-data-export-{user['id']}.json\"",
        },
    )


class DataDeleteBody(BaseModel):
    # Defensive: require the user to type their email to confirm. Prevents
    # accidental deletion from a stale session in a shared browser.
    confirm_email: EmailStr


@router.post("/data-delete")
def data_delete(
    body: DataDeleteBody,
    request: Request,
    response: Response,
    vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
):
    """Soft-delete the current user's account.

    DPDP §12 right to erasure. We mark deleted_at immediately (all access
    blocked from this moment); a separate scheduled job is the right place
    to hard-scrub rows after a 30-day grace window. The current session
    cookie is cleared so the user can't accidentally keep using the SPA.
    """
    from . import dpdp
    user = _require_user(vaaani_session)
    if body.confirm_email.strip().lower() != user["email"].strip().lower():
        raise HTTPException(
            status_code=400,
            detail="confirm_email does not match the signed-in user's email.",
        )
    ip = request.client.host if request.client else None
    dpdp.soft_delete_user(user["id"], ip=ip)
    response.delete_cookie(COOKIE_NAME, **cookie_delete_kwargs())
    return {
        "ok": True,
        "deleted_at": dpdp._now_iso(),
        "message": (
            "Your account is soft-deleted: all access is blocked immediately. "
            "Stored rows are scheduled for hard deletion within 30 days. "
            "Contact iamanushka32@gmail.com within this window to restore."
        ),
    }


@router.get("/dpdp/audit")
def dpdp_audit(vaaani_session: Optional[str] = Cookie(default=None, alias=COOKIE_NAME)):
    """User-visible audit log of DPDP-relevant events on their account."""
    from . import dpdp
    user = _require_user(vaaani_session)
    return {"events": dpdp.audit_history(user["id"], limit=200)}
