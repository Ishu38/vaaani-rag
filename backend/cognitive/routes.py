"""Cognitive routes — FastAPI router for fingerprint, events, and remediation endpoints."""

from fastapi import APIRouter, Cookie, HTTPException
from pydantic import BaseModel

from .store import store
from .fingerprint import build_fingerprint
from .classifier import classify_error, ErrorDiagnosis
from .remediation import generate_remediation

router = APIRouter(prefix="/cognitive", tags=["cognitive"])


class AnalyzeRequest(BaseModel):
    query: str
    student_answer: str
    correct_answer: str
    topic: str = ""
    confidence_1to5: int = 0
    response_ms: float = 0
    session_id: str = ""


class AnalyzeResponse(BaseModel):
    error_type: str
    explanation: str
    root_cause_topic: str
    remediation: str
    confidence_calibration: str
    error_signature: str


def _get_user_id(cookie: str | None) -> int | None:
    if not cookie:
        return None
    try:
        from auth.security import decode_session
        payload = decode_session(cookie)
        if payload:
            return int(payload["sub"])
    except Exception:
        pass
    return None


@router.post("/analyze")
def cognitive_analyze(
    req: AnalyzeRequest,
    vaaani_session: str | None = Cookie(default=None, alias="vaaani_session"),
) -> AnalyzeResponse:
    """Analyze a student answer for cognitive error patterns."""
    user_id = _get_user_id(vaaani_session) or 0
    diagnosis = classify_error(
        query=req.query,
        student_answer=req.student_answer,
        correct_answer=req.correct_answer,
        topic=req.topic,
        confidence_1to5=req.confidence_1to5,
        response_ms=req.response_ms,
        is_correct=req.student_answer.strip().lower() == req.correct_answer.strip().lower(),
    )

    if not diagnosis.is_valid:
        from .detector import quick_coarse_check
        coarse = quick_coarse_check(
            req.student_answer, req.correct_answer,
            req.response_ms, req.confidence_1to5,
        )
        if coarse.is_valid:
            diagnosis = coarse

    if not diagnosis.remediation:
        diagnosis.remediation = generate_remediation(diagnosis.primary_error, req.topic)

    if diagnosis.is_valid and diagnosis.primary_error != "no_error":
        try:
            from .store import CognitiveEvent
            store.log_event(CognitiveEvent(
                user_id=user_id, topic=req.topic, query=req.query,
                student_answer=req.student_answer, correct_answer=req.correct_answer,
                error_type=diagnosis.primary_error,
                error_signature=diagnosis.error_signature,
                explanation=diagnosis.explanation,
                root_cause_topic=diagnosis.root_cause_topic,
                remediation=diagnosis.remediation,
                response_ms=req.response_ms,
                confidence_1to5=req.confidence_1to5,
                actual_correct=1 if req.student_answer.strip().lower() == req.correct_answer.strip().lower() else 0,
                session_id=req.session_id,
            ))
        except Exception:
            pass

    return AnalyzeResponse(
        error_type=diagnosis.primary_error,
        explanation=diagnosis.explanation,
        root_cause_topic=diagnosis.root_cause_topic,
        remediation=diagnosis.remediation,
        confidence_calibration=diagnosis.confidence_calibration,
        error_signature=diagnosis.error_signature,
    )


@router.get("/fingerprint")
def cognitive_fingerprint(
    vaaani_session: str | None = Cookie(default=None, alias="vaaani_session"),
):
    """Get the complete cognitive fingerprint for the current user."""
    user_id = _get_user_id(vaaani_session)
    if not user_id:
        raise HTTPException(401, "Sign in to view cognitive profile")
    fp = build_fingerprint(user_id)
    try:
        store.save_fingerprint(user_id, fp)
    except Exception:
        pass
    return fp


@router.get("/events")
def cognitive_events(
    vaaani_session: str | None = Cookie(default=None, alias="vaaani_session"),
    limit: int = 50,
):
    """Get recent cognitive diagnostic events."""
    user_id = _get_user_id(vaaani_session)
    if not user_id:
        raise HTTPException(401, "Sign in required")
    return {"events": store.get_recent_events(user_id, limit)}


@router.get("/breakdown")
def cognitive_breakdown(
    vaaani_session: str | None = Cookie(default=None, alias="vaaani_session"),
):
    """Get error type breakdown for the user."""
    user_id = _get_user_id(vaaani_session)
    if not user_id:
        raise HTTPException(401, "Sign in required")
    errors = store.error_breakdown(user_id)
    calibration = store.confidence_calibration(user_id)
    topics = store.topic_weakness_map(user_id)
    return {"errors": errors, "calibration": calibration, "topic_weaknesses": topics}


@router.get("/remediation/{error_type}")
def cognitive_remediation(error_type: str, topic: str = ""):
    """Get a remediation suggestion for a specific error type."""
    return {"error_type": error_type, "remediation": generate_remediation(error_type, topic)}
