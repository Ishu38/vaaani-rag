"""Real-time cognitive detector — no LLM. Diagnoses errors and bridges
to the unified Evidence Object Graph so the twin learns from every turn."""

import asyncio
from dataclasses import dataclass, field

from .classifier import classify_error, quick_coarse_check, ErrorDiagnosis
from .store import store, CognitiveEvent


@dataclass
class TurnAnalysis:
    diagnosis: ErrorDiagnosis = field(default_factory=ErrorDiagnosis)
    should_remediate: bool = False
    xray_insight: str = ""


def analyze_turn(
    user_id: int,
    query: str,
    student_answer: str,
    correct_answer: str,
    topic: str = "",
    confidence_1to5: int = 0,
    confidence_0to100: int | None = None,
    response_ms: float = 0,
    is_correct: bool = False,
    session_id: str = "",
    node_id: str = "",
) -> TurnAnalysis:
    analysis = TurnAnalysis()

    if not student_answer.strip():
        return analysis

    # Step 1: quick heuristic check
    coarse = quick_coarse_check(
        student_answer, correct_answer, response_ms, confidence_1to5
    )
    if coarse.is_valid and coarse.primary_error != "no_error":
        analysis.diagnosis = coarse
        analysis.should_remediate = True
        analysis.xray_insight = coarse.explanation

    # Step 2: deterministic deeper classification (no LLM)
    if not analysis.diagnosis.is_valid or (
        not is_correct and analysis.diagnosis.primary_error == "no_error"
    ):
        diagnosis = classify_error(
            query=query,
            student_answer=student_answer,
            correct_answer=correct_answer,
            topic=topic,
            confidence_1to5=confidence_1to5,
            response_ms=response_ms,
            is_correct=is_correct,
        )
        if diagnosis.is_valid:
            analysis.diagnosis = diagnosis
            analysis.should_remediate = diagnosis.primary_error != "no_error"
            analysis.xray_insight = diagnosis.explanation

    # Step 3: log to cognitive event store (existing path)
    if analysis.diagnosis.is_valid:
        event = CognitiveEvent(
            user_id=user_id,
            topic=topic,
            query=query,
            student_answer=student_answer,
            correct_answer=correct_answer,
            error_type=analysis.diagnosis.primary_error,
            error_signature=analysis.diagnosis.error_signature,
            explanation=analysis.diagnosis.explanation,
            root_cause_topic=analysis.diagnosis.root_cause_topic,
            remediation=analysis.diagnosis.remediation,
            response_ms=response_ms,
            confidence_1to5=confidence_1to5,
            actual_correct=1 if is_correct else 0,
            session_id=session_id,
        )
        try:
            store.log_event(event)
        except Exception:
            pass

    # Step 4: bridge to Evidence Object Graph (new path — unify databases)
    if node_id:
        try:
            from evidence_graph import EvidenceObject
            import cognitive_twin as twin
            sid = str(user_id)
            outcome = "correct" if is_correct else "incorrect"
            ev = EvidenceObject(
                sid, node_id, "quiz", outcome,
                confidence=confidence_1to5 / 5.0 if confidence_1to5 else 0.8,
                meta={
                    "error_type": analysis.diagnosis.primary_error,
                    "confidence_1to5": confidence_1to5,
                    "response_ms": response_ms,
                },
            )
            twin.update(ev)
        except Exception:
            pass

    # Always log confidence
    try:
        store.log_confidence(
            user_id=user_id,
            topic=topic,
            query=query,
            answer=student_answer,
            confidence_1to5=confidence_1to5,
            actual_correct=1 if is_correct else 0,
            response_ms=response_ms,
        )
    except Exception:
        pass

    return analysis


async def analyze_turn_async(
    user_id: int,
    query: str,
    student_answer: str,
    correct_answer: str,
    topic: str = "",
    confidence_1to5: int = 0,
    response_ms: float = 0,
    is_correct: bool = False,
    session_id: str = "",
    node_id: str = "",
) -> TurnAnalysis:
    """Async wrapper — use in FastAPI endpoints to avoid blocking."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        analyze_turn,
        user_id, query, student_answer, correct_answer,
        topic, confidence_1to5, response_ms, is_correct, session_id,
        node_id,
    )
