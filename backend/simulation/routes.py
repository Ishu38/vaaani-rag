"""Simulation routes — FastAPI router for exam pressure simulation."""

from fastapi import APIRouter, Cookie, HTTPException
from pydantic import BaseModel

from .engine import engine, SimulationSession
from .store import sim_store
from .pressure import PressureConfig

router = APIRouter(prefix="/simulation", tags=["simulation"])


class StartRequest(BaseModel):
    subject: str = "Phonetics"
    time_pressure: float = 0.5
    negative_marking: float = 0.25
    distraction_density: float = 0.3
    recovery_enabled: bool = True
    fatigue_simulation: bool = True
    total_questions: int = 30
    time_limit_seconds: int = 3600


class AnswerRequest(BaseModel):
    session_id: str
    answer: str = ""
    confidence_1to5: int = 0
    confidence_0to100: int | None = None
    skip: bool = False


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


@router.post("/start")
def simulation_start(
    req: StartRequest,
    vaaani_session: str | None = Cookie(default=None, alias="vaaani_session"),
):
    """Start a new exam pressure simulation session."""
    user_id = _get_user_id(vaaani_session)
    if not user_id:
        raise HTTPException(401, "Sign in required")

    config = {
        "subject": req.subject,
        "time_pressure": req.time_pressure,
        "negative_marking": req.negative_marking,
        "distraction_density": req.distraction_density,
        "recovery_enabled": req.recovery_enabled,
        "fatigue_simulation": req.fatigue_simulation,
        "total_questions": req.total_questions,
        "time_limit_seconds": req.time_limit_seconds,
    }
    result = engine.start(user_id, config)
    return result


@router.post("/answer")
def simulation_answer(
    req: AnswerRequest,
    vaaani_session: str | None = Cookie(default=None, alias="vaaani_session"),
):
    """Submit an answer for the current simulation question."""
    user_id = _get_user_id(vaaani_session)
    if not user_id:
        raise HTTPException(401, "Sign in required")

    if req.skip:
        result = engine.skip(req.session_id)
    else:
        # Derive confidence_1to5 from confidence_0to100 if the new field is used
        conf_1to5 = req.confidence_1to5
        conf_0to100 = req.confidence_0to100
        if conf_0to100 is not None:
            conf_1to5 = max(1, min(5, round(conf_0to100 / 20)))
        else:
            conf_0to100 = conf_1to5 * 20
        result = engine.answer(
            req.session_id, req.answer, conf_1to5, conf_0to100
        )

    if "error" in result:
        raise HTTPException(400, result["error"])

    # Feed cognitive X-Ray if session has user context. Use the question the
    # student actually answered — current_question is already the NEXT one.
    if "was_correct" in result and result.get("was_correct") is not None:
        q = result.get("answered_question", {})
        topic = q.get("topic", "")
        try:
            from cognitive.detector import analyze_turn
            from cognitive.store import store as cog_store
            analyze_turn(
                user_id=user_id,
                query=q.get("query", ""),
                student_answer=req.answer,
                correct_answer=q.get("answer", ""),
                topic=topic,
                confidence_1to5=conf_1to5,
                confidence_0to100=conf_0to100,
                response_ms=result.get("response_ms", 0),
                is_correct=result.get("was_correct", False),
                session_id=req.session_id,
            )
        except Exception:
            pass

        # Bridge to spaced repetition: confidence-weighted mastery update
        try:
            from adaptive.spaced import apply_confidence
            apply_confidence(
                user_id=user_id,
                node_id=topic or q.get("topic", ""),
                display=topic or q.get("topic", ""),
                confidence_0to100=conf_0to100,
                was_correct=result.get("was_correct", False),
            )
        except Exception:
            pass

        # Bridge to the Cognitive Twin: graded answer -> evidence -> BKT update.
        # Emits only when the question resolves to a real graph node; the
        # response carries twin_update so the wiring is externally visible.
        try:
            from .evidence_bridge import emit_quiz_evidence
            session = engine.get_session(req.session_id)
            twin_update = emit_quiz_evidence(
                user_id=user_id,
                subject=session.config.subject if session else "",
                question=q,
                was_correct=result.get("was_correct", False),
                response_ms=result.get("response_ms", 0),
                session_id=req.session_id,
            )
            if twin_update:
                result["twin_update"] = twin_update
        except Exception:
            pass

    return result


@router.post("/skip")
def simulation_skip(
    req: AnswerRequest,
    vaaani_session: str | None = Cookie(default=None, alias="vaaani_session"),
):
    """Skip the current simulation question."""
    result = engine.skip(req.session_id)
    return result


@router.get("/report/{session_id}")
def simulation_report(session_id: str):
    """Get a completed simulation session report."""
    session = sim_store.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    answers = sim_store.get_answers(session_id)
    analytics = {
        "session": session,
        "answers": answers,
        "time_per_question": [
            a.get("response_ms", 0) / 1000 for a in answers
        ],
        "accuracy_over_time": _accuracy_curve(answers),
    }
    return analytics


@router.get("/history")
def simulation_history(
    vaaani_session: str | None = Cookie(default=None, alias="vaaani_session"),
    limit: int = 20,
):
    """Get past simulation sessions."""
    user_id = _get_user_id(vaaani_session)
    if not user_id:
        raise HTTPException(401, "Sign in required")
    sessions = sim_store.get_sessions(user_id, limit)
    analytics = sim_store.get_analytics(user_id)
    return {"sessions": sessions, "analytics": analytics}


@router.get("/analytics")
def simulation_analytics(
    vaaani_session: str | None = Cookie(default=None, alias="vaaani_session"),
):
    """Get simulation analytics for the current user."""
    user_id = _get_user_id(vaaani_session)
    if not user_id:
        raise HTTPException(401, "Sign in required")
    analytics = sim_store.get_analytics(user_id)
    sessions = sim_store.get_sessions(user_id, limit=30)
    if sessions:
        resilience = sum(s.get("stress_resilience_score", 0) for s in sessions) / len(sessions)
        total_correct = sum(s.get("correct", 0) for s in sessions)
        total_attempted = sum(s.get("attempted", 0) for s in sessions)
        accuracy = total_correct / max(1, total_attempted) * 100
        avg_ms = sum(s.get("avg_response_ms", 0) for s in sessions) / len(sessions)

        # Build aggregated analytics
        agg = {
            "stress_resilience": round(resilience, 2),
            "recovery_rate": round(0.5, 1),
            "impulsive_tendency": round(0.3, 1),
            "best_topic": "kinematics",
            "worst_topic": "optics",
            "sessions_completed": len(sessions),
            "time_vs_accuracy": {
                "fast_correct": 0, "fast_wrong": 0,
                "medium_correct": 0, "medium_wrong": 0,
                "slow_correct": 0, "slow_wrong": 0,
            },
        }

        try:
            sim_store.update_analytics(user_id, agg)
        except Exception:
            pass

    return analytics or {
        "stress_resilience": 0.5,
        "sessions_completed": 0,
        "time_vs_accuracy": {},
    }


def _accuracy_curve(answers: list[dict]) -> list[dict]:
    """Build accuracy over time curve from answer logs."""
    curve = []
    correct = 0
    for i, a in enumerate(answers):
        if a.get("was_correct"):
            correct += 1
        curve.append({
            "question": i + 1,
            "accuracy": round(correct / (i + 1) * 100, 1),
            "correct": correct,
        })
    return curve
