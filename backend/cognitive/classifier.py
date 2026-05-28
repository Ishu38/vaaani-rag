"""Error classifier — uses DeepSeek to diagnose cognitive error patterns."""

import json
import hashlib
from dataclasses import dataclass, field

from config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_BASE_URL, DEEPSEEK_TIMEOUT
import httpx


ERROR_TYPES = [
    "memorization_override",
    "conceptual_gap",
    "algebraic_slip",
    "dimensional_error",
    "overconfidence",
    "underconfidence",
    "impulsive",
    "shortcut_dependency",
    "fragile_understanding",
    "visualization_weakness",
    "unit_confusion",
    "sign_error",
    "no_error",
]


CLASSIFIER_SYSTEM = (
    "You are a cognitive diagnostic AI for JEE exam preparation. "
    "Your job is to diagnose *why* a student got an answer wrong, not just label it incorrect. "
    "Diagnose the thinking pattern, not the content. "
    "Respond with VALID JSON only, no markdown fences."
)

CLASSIFIER_TEMPLATE = """Analyze the student's answer for cognitive error patterns.

PROBLEM: {query}

STUDENT'S ANSWER: {student_answer}

CORRECT ANSWER: {correct_answer}

CONTEXT: {topic}

ADDITIONAL DATA:
- Confidence (1-5): {confidence}
- Response time: {response_ms:.0f} ms
- Is the answer correct? {is_correct}

Classify the PRIMARY error type from this list:
{error_types_list}

Output JSON:
{{
    "primary_error": "one of the error types above, or 'no_error' if correct",
    "explanation": "human-readable diagnosis of the thinking pattern. Be specific — mention what the student is likely doing wrong in their head.",
    "root_cause_topic": "the foundational concept that is actually weak, e.g. 'trigonometric identities' rather than just 'calculus'",
    "remediation": "specific next step for the student, e.g. 'Revisit the geometric interpretation of sin²x+cos²x=1 with unit circle diagrams'",
    "confidence_calibration": "well_calibrated|overconfident|underconfident",
    "error_signature": "a compact hashable signature like conceptual_gap::trig_identity::additive_bias"
}}"""


@dataclass
class ErrorDiagnosis:
    primary_error: str = "no_error"
    explanation: str = ""
    root_cause_topic: str = ""
    remediation: str = ""
    confidence_calibration: str = "well_calibrated"
    error_signature: str = ""
    is_valid: bool = False


def _signature_hash(sig: str) -> str:
    return hashlib.sha1(sig.encode()).hexdigest()[:12]


def classify_error(
    query: str,
    student_answer: str,
    correct_answer: str,
    topic: str = "",
    confidence_1to5: int = 0,
    response_ms: float = 0,
    is_correct: bool = False,
) -> ErrorDiagnosis:
    if not DEEPSEEK_API_KEY:
        return ErrorDiagnosis()

    if is_correct and (not confidence_1to5 or confidence_1to5 > 2):
        return ErrorDiagnosis(primary_error="no_error", is_valid=True)

    prompt = CLASSIFIER_TEMPLATE.format(
        query=query,
        student_answer=student_answer,
        correct_answer=correct_answer,
        topic=topic,
        confidence=confidence_1to5,
        response_ms=response_ms,
        is_correct="yes" if is_correct else "no",
        error_types_list=", ".join(ERROR_TYPES),
    )

    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    try:
        r = httpx.post(
            f"{DEEPSEEK_BASE_URL.rstrip('/')}/v1/chat/completions",
            json={
                "model": DEEPSEEK_MODEL,
                "messages": messages,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                     "Content-Type": "application/json"},
            timeout=DEEPSEEK_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        result = json.loads(content)
    except Exception:
        return ErrorDiagnosis()

    diagnosis = ErrorDiagnosis(
        primary_error=result.get("primary_error", "no_error"),
        explanation=result.get("explanation", ""),
        root_cause_topic=result.get("root_cause_topic", ""),
        remediation=result.get("remediation", ""),
        confidence_calibration=result.get("confidence_calibration", "well_calibrated"),
        error_signature=result.get("error_signature", ""),
        is_valid=True,
    )

    if diagnosis.primary_error not in ERROR_TYPES:
        diagnosis.primary_error = "conceptual_gap"
        diagnosis.is_valid = True

    return diagnosis


def quick_coarse_check(student_answer: str, correct_answer: str,
                       response_ms: float = 0, confidence_1to5: int = 0) -> ErrorDiagnosis:
    """Fast heuristic before LLM call — catches obvious patterns."""
    sa = student_answer.strip().lower()
    ca = correct_answer.strip().lower()

    if not sa:
        return ErrorDiagnosis(
            primary_error="conceptual_gap",
            explanation="Student left answer blank.",
            root_cause_topic="unknown",
            remediation="Build confidence with simpler warm-up problems before full difficulty.",
            confidence_calibration="underconfident",
        )

    if sa == ca:
        return ErrorDiagnosis(primary_error="no_error", is_valid=True)

    # Algebraic slip: answers differ only by sign
    if sa.replace("-", "") == ca.replace("-", ""):
        return ErrorDiagnosis(
            primary_error="sign_error",
            explanation="Approach is correct but sign is flipped.",
            root_cause_topic="algebraic_manipulation",
            remediation="Double-check sign conventions, especially with negative numbers and subtraction.",
            error_signature="sign_error::sign_flip",
        )

    # Dimensional: answers off by factor of 10, 100, 1000
    for scale in [10, 100, 1000]:
        try:
            if abs(float(sa) - float(ca) * scale) < 0.01:
                return ErrorDiagnosis(
                    primary_error="dimensional_error",
                    explanation=f"Answer is off by factor of {scale} — likely unit conversion error.",
                    root_cause_topic="dimensional_analysis",
                    remediation="Practice unit conversion and dimensional analysis.",
                    error_signature=f"dimensional_error::off_by_{scale}",
                )
        except ValueError:
            pass

    # Impulsive: very fast wrong answer (answers don't match and it was fast)
    is_wrong = student_answer != correct_answer
    if response_ms > 0 and response_ms < 10000 and is_wrong:
        return ErrorDiagnosis(
            primary_error="impulsive",
            explanation=f"Answered in {response_ms/1000:.1f}s — likely impulsive without verification.",
            root_cause_topic="metacognition",
            remediation="Slow down. Count to 3 before submitting. Verify at least one check.",
            error_signature="impulsive::fast_wrong",
        )

    return ErrorDiagnosis()  # Needs LLM classification
