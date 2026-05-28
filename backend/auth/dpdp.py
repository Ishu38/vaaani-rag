"""DPDP Act 2023 scaffolding — parental consent + data subject rights.

This module implements the technical layer for India's Digital Personal Data
Protection Act 2023 §9 (children's data) and §11-12 (right to access /
right to erase). The verifiable-parent-identity piece (Aadhaar / DigiLocker)
is INTENTIONALLY NOT WIRED — see [[feedback_vaaani_cf_streaming]] and the
DPDP scoping doc. Email-magic-link consent satisfies the spirit but not the
final Rules' letter; ship as "DPDP-aware", not "DPDP-certified".
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional

from config import APP_BASE_URL
from .db import connect
from .email_sender import get_sender as get_email_sender
from .security import random_token

# Consent text version — bump whenever the legal text changes so audit logs
# show exactly which version each parent agreed to.
CONSENT_TEXT_VERSION = "v1.0.1-2026-05-26"  # bumped: grievance contact updated to neilshankarray@vaaani.in

CONSENT_TEXT = """\
Vaaani is a Socratic study assistant for students. Under India's Digital
Personal Data Protection Act 2023, your child needs your verifiable consent
before we may process their personal data.

WHAT WE COLLECT FROM YOUR CHILD:
  • Account details: name, email, date of birth, school (if applicable)
  • Study activity: questions asked, documents uploaded, ratings given
  • Spaced-review state: which topics they've practised and how well

WHY WE PROCESS IT:
  • To run the study assistant (answers, retrieval, audio narration)
  • To track mastery over time (per-topic skill model + spaced review)
  • To enforce per-school guardrails when your child belongs to a licensed
    school org (curriculum scope, Socratic-only mode, etc.)

WHERE IT IS STORED:
  • Account + study data: encrypted disk on a managed VPS
  • Question content is sent to DeepSeek (an LLM provider) to generate
    answers. DeepSeek may temporarily process the text outside India.
  • You may withdraw consent at any time, after which we will stop
    processing your child's data and (on request) delete it.

YOUR RIGHTS AS PARENT (DPDP §11-§14):
  • Right to access your child's stored data
  • Right to correct or erase it
  • Right to withdraw this consent at any time
  • Right to nominate a successor data principal
  • Right to grievance redressal (contact: neilshankarray@vaaani.in)

By clicking CONFIRM you state that:
  1. You are the parent or lawful guardian of this child
  2. You have read and understood the above
  3. You consent to Vaaani processing your child's data for the purposes
     listed, until you withdraw consent

This consent record is stored with your IP address, the timestamp, and the
version of this notice you agreed to (currently """ + CONSENT_TEXT_VERSION + """).
"""

CONSENT_TOKEN_EXP_HOURS = 168  # 7 days


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def age_from_dob(dob_iso: str) -> Optional[int]:
    """Compute completed years between dob_iso (YYYY-MM-DD) and today.

    Returns None if dob_iso isn't a parseable ISO date; caller treats that
    as 'unknown age' and applies the safest policy (require consent).
    """
    if not dob_iso:
        return None
    try:
        dob = date.fromisoformat(dob_iso)
    except ValueError:
        return None
    today = date.today()
    years = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return max(0, years)


def is_minor(dob_iso: Optional[str]) -> bool:
    """Under-18 by DPDP §9 definition. Treats unknown DOB as minor (safer)."""
    if not dob_iso:
        return True
    age = age_from_dob(dob_iso)
    return age is None or age < 18


# ---------------- audit log ----------------

def audit(
    user_id: Optional[int],
    event_type: str,
    detail: str = "",
    *,
    ip: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> None:
    """Append-only event log. Failures swallowed — never break the request."""
    try:
        with connect() as c:
            c.execute(
                "INSERT INTO dpdp_audit_log (user_id, event_type, detail, actor_ip, actor_user_agent) VALUES (?,?,?,?,?)",
                (user_id, event_type, detail[:1000], ip, (user_agent or "")[:500]),
            )
    except Exception:
        pass


def audit_history(user_id: int, limit: int = 100) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT id, event_type, detail, actor_ip, created_at FROM dpdp_audit_log "
            "WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------- consent issue / verify / withdraw ----------------

def request_parental_consent(
    child_user_id: int,
    parent_email: str,
    parent_name: Optional[str] = None,
    parent_phone: Optional[str] = None,
    *,
    actor_ip: Optional[str] = None,
) -> dict:
    """Create a consent record, persist the token, dispatch the parent email.

    Idempotent-ish: if a pending consent already exists for this child, it
    is superseded (new token, new audit row). The expired token simply rots.
    """
    parent_email_norm = parent_email.strip().lower()
    if not parent_email_norm or "@" not in parent_email_norm:
        raise ValueError("Parent email is required.")

    token = random_token(24)
    expires_at = (
        datetime.now(timezone.utc) + timedelta(hours=CONSENT_TOKEN_EXP_HOURS)
    ).isoformat()
    with connect() as c:
        c.execute(
            """INSERT INTO parental_consents
               (child_user_id, parent_name, parent_email, parent_phone,
                consent_token, consent_text_version, expires_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                child_user_id, parent_name, parent_email_norm, parent_phone,
                token, CONSENT_TEXT_VERSION, expires_at,
            ),
        )
        consent_id = c.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
        c.execute(
            "UPDATE users SET consent_status = 'pending', active_consent_id = ? WHERE id = ?",
            (consent_id, child_user_id),
        )

    audit(
        child_user_id, "consent_requested",
        f"parent={parent_email_norm} consent_id={consent_id} version={CONSENT_TEXT_VERSION}",
        ip=actor_ip,
    )

    link = f"{APP_BASE_URL.rstrip('/')}/parental-consent?token={token}"
    body_text = (
        f"Hello,\n\n"
        f"Your child (account email registered with Vaaani) has signed up for "
        f"Vaaani — a Socratic AI study assistant. Indian law (DPDP Act 2023, §9) "
        f"requires your verifiable consent before we may process their data.\n\n"
        f"Please review what we collect, why, and your rights at this link:\n"
        f"{link}\n\n"
        f"The link expires in 7 days. If you did not expect this email, you may "
        f"ignore it — your child's account remains inactive until you confirm."
    )
    body_html = (
        f"<p>Hello,</p>"
        f"<p>Your child has signed up for <strong>Vaaani</strong> — a Socratic AI "
        f"study assistant. Indian law (<em>Digital Personal Data Protection Act 2023, §9</em>) "
        f"requires your verifiable consent before we may process their data.</p>"
        f"<p>Please review what we collect, why, and your rights here:</p>"
        f"<p><a href='{link}' style='background:#4d694e;color:#fff;padding:10px 18px;"
        f"border-radius:6px;text-decoration:none;display:inline-block;'>"
        f"Review and confirm consent</a></p>"
        f"<p>The link expires in 7 days. If you did not expect this email, you may "
        f"ignore it — your child's account remains inactive until you confirm.</p>"
        f"<p style='color:#666;font-size:12px;'>Consent text version "
        f"{CONSENT_TEXT_VERSION}. Withdraw anytime from your child's account page "
        f"or by emailing neilshankarray@vaaani.in.</p>"
    )
    try:
        get_email_sender().send(
            parent_email_norm,
            "Action required: parental consent for your child's Vaaani account",
            body_text, body_html,
        )
    except Exception as e:
        audit(child_user_id, "consent_email_failed", str(e)[:300])
        # Don't raise — caller still sees consent record created; they can
        # trigger a resend via the consent_status endpoint.

    return {
        "consent_id": consent_id,
        "parent_email": parent_email_norm,
        "consent_text_version": CONSENT_TEXT_VERSION,
        "expires_at": expires_at,
    }


def lookup_consent_by_token(token: str) -> Optional[dict]:
    if not token:
        return None
    with connect() as c:
        row = c.execute(
            "SELECT * FROM parental_consents WHERE consent_token = ?", (token,),
        ).fetchone()
        if not row:
            return None
        if row["granted_at"] or row["withdrawn_at"]:
            return dict(row)  # still return so the page can show status
        if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
            return dict(row)
        return dict(row)


def confirm_consent(
    token: str,
    parent_name: Optional[str] = None,
    *,
    ip: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> dict:
    """Mark a consent record as granted. Returns the updated consent dict."""
    row = lookup_consent_by_token(token)
    if not row:
        raise ValueError("Invalid consent link.")
    if row["granted_at"]:
        return row  # already granted — idempotent
    if row["withdrawn_at"]:
        raise ValueError("This consent has already been withdrawn.")
    if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
        raise ValueError("This consent link has expired. Ask your child to request a new one.")

    now = _now_iso()
    with connect() as c:
        c.execute(
            """UPDATE parental_consents
               SET granted_at = ?, parent_ip = ?, parent_user_agent = ?,
                   parent_name = COALESCE(?, parent_name)
               WHERE id = ?""",
            (now, ip, (user_agent or "")[:500], parent_name, row["id"]),
        )
        c.execute(
            "UPDATE users SET consent_status = 'granted' WHERE id = ?",
            (row["child_user_id"],),
        )

    audit(
        row["child_user_id"], "consent_granted",
        f"consent_id={row['id']} version={row['consent_text_version']} parent={row['parent_email']}",
        ip=ip, user_agent=user_agent,
    )
    return lookup_consent_by_token(token)


def withdraw_consent(
    consent_id: int,
    *,
    actor_user_id: Optional[int] = None,
    ip: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> dict:
    """Mark a consent as withdrawn; flip the child to consent_status='withdrawn'.

    Withdrawal IMMEDIATELY blocks processing (the gate predicate refuses any
    status that isn't 'granted' or 'not_required'). Data is not erased here —
    a separate erasure request (right-to-erase) is the user's other lever.
    """
    now = _now_iso()
    with connect() as c:
        row = c.execute(
            "SELECT * FROM parental_consents WHERE id = ?", (consent_id,)
        ).fetchone()
        if not row:
            raise ValueError("Consent record not found.")
        if row["withdrawn_at"]:
            return dict(row)
        c.execute(
            "UPDATE parental_consents SET withdrawn_at = ? WHERE id = ?",
            (now, consent_id),
        )
        c.execute(
            "UPDATE users SET consent_status = 'withdrawn' WHERE id = ?",
            (row["child_user_id"],),
        )
    audit(
        row["child_user_id"], "consent_withdrawn",
        f"consent_id={consent_id} actor={actor_user_id}",
        ip=ip, user_agent=user_agent,
    )
    with connect() as c:
        return dict(
            c.execute("SELECT * FROM parental_consents WHERE id = ?", (consent_id,)).fetchone()
        )


def consent_status_for_user(user: dict) -> dict:
    """Return the consent state needed by the SPA + gate predicate."""
    dob = user.get("date_of_birth")
    minor = is_minor(dob)
    status = user.get("consent_status") or "not_required"
    active_id = user.get("active_consent_id")
    consent_row = None
    if active_id:
        with connect() as c:
            row = c.execute(
                "SELECT id, parent_email, parent_name, requested_at, granted_at, "
                "withdrawn_at, consent_text_version, expires_at FROM parental_consents "
                "WHERE id = ?", (active_id,),
            ).fetchone()
            if row:
                consent_row = dict(row)
    return {
        "is_minor": minor,
        "age": age_from_dob(dob) if dob else None,
        "consent_status": status,
        "needs_consent": minor and status != "granted",
        "active_consent": consent_row,
    }


# ---------------- the gate predicate ----------------

def allow_processing(user: dict) -> tuple[bool, str]:
    """The single source of truth used by every gated endpoint.

    Returns (allowed, reason). reason is empty when allowed; otherwise it's
    a short machine-readable code the frontend uses to route the user
    (e.g. 'consent_required' → modal explaining parent-consent path).
    """
    if user.get("deleted_at"):
        return False, "account_deleted"
    if not is_minor(user.get("date_of_birth")):
        return True, ""
    status = user.get("consent_status") or "not_required"
    if status == "granted":
        return True, ""
    if status == "withdrawn":
        return False, "consent_withdrawn"
    return False, "consent_required"


# ---------------- §11 data export / §12 erasure ----------------

def export_user_data(user_id: int) -> dict:
    """Return everything we hold about this user, in one JSON-serialisable dict.

    Intentionally inclusive — DPDP §11 grants the right to know what's stored.
    Sensitive fields (password hash) are redacted; everything else is included.
    """
    with connect() as c:
        u = c.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not u:
            return {"error": "user not found"}
        user = dict(u)
        user["password_hash"] = "[redacted]"

        def fetch_all(sql: str, params: tuple = ()) -> list[dict]:
            return [dict(r) for r in c.execute(sql, params).fetchall()]

        return {
            "exported_at": _now_iso(),
            "subject": user,
            "school_memberships": fetch_all(
                "SELECT * FROM school_memberships WHERE user_id = ?", (user_id,),
            ),
            "parent_links_as_parent": fetch_all(
                "SELECT * FROM parent_student_links WHERE parent_user_id = ?", (user_id,),
            ),
            "parent_links_as_student": fetch_all(
                "SELECT * FROM parent_student_links WHERE student_user_id = ?", (user_id,),
            ),
            "parental_consents": fetch_all(
                "SELECT * FROM parental_consents WHERE child_user_id = ?", (user_id,),
            ),
            "student_skills": fetch_all(
                "SELECT * FROM student_skills WHERE user_id = ?", (user_id,),
            ),
            "student_attempts": fetch_all(
                "SELECT id, topic, rating, query, at FROM student_attempts WHERE user_id = ?",
                (user_id,),
            ),
            "dpdp_audit_log": fetch_all(
                "SELECT id, event_type, detail, created_at FROM dpdp_audit_log WHERE user_id = ?",
                (user_id,),
            ),
            "notice": (
                "This export contains only data stored in the Vaaani users database. "
                "It does NOT include: free-form chat transcripts (not retained by "
                "default), document chunks you uploaded (those live in the vector "
                "index keyed by content hash, not by user). To erase ingested "
                "documents, use the account-deletion endpoint after exporting."
            ),
        }


def soft_delete_user(user_id: int, *, ip: Optional[str] = None) -> None:
    """Mark a user as deleted: blocks all access immediately.

    Hard-row scrubbing is intentionally deferred — gives a 30-day grace window
    for accidental deletions to be reversed by support, and lets us preserve
    school audit logs that reference the user_id. A separate batch job
    (not wired here) is the right place to actually drop rows after 30 days.
    """
    now = _now_iso()
    with connect() as c:
        c.execute("UPDATE users SET deleted_at = ? WHERE id = ?", (now, user_id))
        # Cascade: revoke active consents so 'granted' state can't be reused
        # if the row is ever restored.
        c.execute(
            "UPDATE parental_consents SET withdrawn_at = COALESCE(withdrawn_at, ?) "
            "WHERE child_user_id = ? AND granted_at IS NOT NULL AND withdrawn_at IS NULL",
            (now, user_id),
        )
    audit(user_id, "data_deleted", "soft_delete; 30-day scrub pending", ip=ip)
