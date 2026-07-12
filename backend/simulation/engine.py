"""Simulation engine — state machine that drives exam pressure simulation."""

import re
import uuid
import time
from dataclasses import dataclass, field
from enum import Enum

from .pressure import PressureController, PressureConfig, PressureState, Phase
from .coach import CoachInterjector
from .question_bank import QuestionBank
from .store import sim_store


_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation/quotes/hyphens, map number words, strip articles."""
    t = (text or "").lower()
    t = re.sub(r"[^\w\s]", " ", t)
    words = [w for w in t.split() if w not in ("a", "an", "the")]
    words = [_NUMBER_WORDS.get(w, w) for w in words]
    return " ".join(words)


def _grade(student_answer: str, q: dict) -> bool:
    """Lenient free-text grading for linguistics answers.

    Correct when the normalized student answer equals the key (with any
    parenthetical elaboration stripped), or contains it, or contains any of
    the question's curated `accept` phrases. Exact string equality alone
    marks almost every honest free-text answer wrong.
    """
    student = _normalize(student_answer)
    if not student:
        return False
    key = q.get("answer", "")
    key_short = re.sub(r"\([^)]*\)", " ", key)  # drop parenthetical elaboration
    candidates = [key, key_short] + list(q.get("accept", []) or [])
    for cand in candidates:
        c = _normalize(cand)
        if not c or (len(c) < 2 and not c.isdigit()):
            continue
        if student == c:
            return True
        # Containment with word boundaries: the key phrase appears inside a
        # longer honest answer ("it is a nasal sound" matches key "nasal"),
        # but "ong" cannot match inside "strong".
        if re.search(rf"(?<!\w){re.escape(c)}(?!\w)", student):
            return True
    return False


class SessionState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STRESS_SPIKE = "stress_spike"
    RECOVERY = "recovery"
    FINAL = "final"
    COMPLETE = "complete"


@dataclass
class SimulationSession:
    session_id: str
    user_id: int
    config: PressureConfig
    controller: PressureController
    question_bank: QuestionBank = field(default_factory=QuestionBank)
    coach: CoachInterjector = field(default_factory=CoachInterjector)
    state: str = "idle"
    started_at: float = 0
    current_question: dict | None = None
    question_index: int = 0
    total_questions: int = 0


class SimulationEngine:
    _sessions: dict[str, SimulationSession] = {}

    def start(self, user_id: int, config: dict) -> dict:
        pc = PressureConfig(
            time_pressure=config.get("time_pressure", 0.5),
            negative_marking=config.get("negative_marking", 0.25),
            distraction_density=config.get("distraction_density", 0.3),
            recovery_enabled=config.get("recovery_enabled", True),
            fatigue_simulation=config.get("fatigue_simulation", True),
            total_questions=config.get("total_questions", 30),
            time_limit_seconds=config.get("time_limit_seconds", 3600),
            subject=config.get("subject", "Phonetics"),
        )
        session_id = uuid.uuid4().hex[:16]
        controller = PressureController(pc)
        controller.start()
        qb = QuestionBank(pc.subject)

        session = SimulationSession(
            session_id=session_id,
            user_id=user_id,
            config=pc,
            controller=controller,
            question_bank=qb,
            coach=CoachInterjector(),
            state="running",
            started_at=time.time(),
            total_questions=pc.total_questions,
        )
        self._sessions[session_id] = session

        try:
            sim_store.create_session(session_id, user_id, pc.subject, config)
        except Exception:
            pass

        # First question
        q = qb.get_question(controller.state, question_index=0)
        session.current_question = q
        session.question_index = 0
        controller.question_start_time = time.time()

        return self._session_response(session, q)

    def answer(self, session_id: str, answer: str, confidence_1to5: int = 0,
               confidence_0to100: int = -1) -> dict:
        session = self._sessions.get(session_id)
        if not session:
            return {"error": "session not found", "state": "unknown"}
        if session.state == "complete":
            return {"error": "session completed", "state": "complete"}

        q = session.current_question
        if not q:
            return {"error": "no active question"}

        response_ms = (time.time() - session.controller.question_start_time) * 1000
        correct = _grade(answer, q)

        session.controller.on_answer(correct, response_ms < 15000, response_ms)
        # Tick down by the time spent on THIS question only — passing total
        # session elapsed here double-counts and ends sessions early.
        session.controller.tick(response_ms / 1000.0)

        # Log
        try:
            sim_store.log_answer(
                session_id=session_id,
                user_id=session.user_id,
                question_index=session.question_index,
                topic=q.get("topic", ""),
                difficulty=session.controller.state.current_difficulty,
                query=q.get("query", ""),
                correct_answer=q.get("answer", ""),
                student_answer=answer,
                was_correct=1 if correct else 0,
                response_ms=response_ms,
                confidence_1to5=confidence_1to5,
                coaching_interjection="",
                pressure_state={
                    "difficulty": session.controller.state.current_difficulty,
                    "phase": session.controller.state.phase,
                    "stress": session.controller.state.stress_level,
                    "momentum": session.controller.state.momentum,
                },
                is_flagged=0,
                confidence_0to100=confidence_0to100 if confidence_0to100 >= 0 else confidence_1to5 * 20,
            )
        except Exception:
            pass

        # Coaching interjection
        coaching = ""
        if session.controller.should_coach():
            coaching = session.coach.generate(
                state=session.controller.state,
                was_correct=correct,
                response_ms=response_ms,
            )
            session.controller.coach_triggered()
            session.controller.recover_from_wrong_streak()

        # ── Confidence-driven adaptive difficulty + metacognitive feedback ──
        # The learner's pre-submission confidence (0–100%) adjusts how the
        # pressure controller responds to the answer and generates
        # metacognitive feedback the frontend displays after each answer.
        conf_feedback = _confidence_feedback(
            confidence_0to100, correct, q.get("topic", ""),
        )
        _apply_confidence_to_difficulty(
            session.controller, correct, confidence_0to100,
        )

        # Next question
        session.question_index += 1
        if session.question_index >= session.config.total_questions or (
            session.controller.state.time_remaining_seconds <= 0
        ):
            result = self._complete(session, coaching, correct, response_ms)
            result["answered_question"] = {
                "query": q.get("query", ""),
                "answer": q.get("answer", ""),
                "topic": q.get("topic", ""),
            }
            return result

        next_q = session.question_bank.get_question(
            session.controller.state,
            question_index=session.question_index,
        )
        session.current_question = next_q
        session.controller.question_start_time = time.time()

        result = self._session_response(session, next_q, coaching, correct, response_ms)
        # The question the student just answered (with its correct answer) —
        # for post-answer feedback and the cognitive X-Ray feed. current_question
        # is already the NEXT question here, so consumers must not use it for that.
        result["answered_question"] = {
            "query": q.get("query", ""),
            "answer": q.get("answer", ""),
            "topic": q.get("topic", ""),
        }
        result["confidence_feedback"] = conf_feedback
        return result

    def skip(self, session_id: str) -> dict:
        session = self._sessions.get(session_id)
        if not session:
            return {"error": "session not found"}
        session.controller.tick(time.time() - session.controller.question_start_time)
        session.controller.on_skip()
        session.question_index += 1
        if session.question_index >= session.config.total_questions:
            return self._complete(session, "", False, 0)
        next_q = session.question_bank.get_question(
            session.controller.state, question_index=session.question_index
        )
        session.current_question = next_q
        session.controller.question_start_time = time.time()
        return self._session_response(session, next_q)

    def _session_response(
        self, session: SimulationSession, question: dict,
        coaching: str = "", correct: bool | None = None, response_ms: float = 0
    ) -> dict:
        ps = session.controller.state
        # Never ship the correct answer (or accepted variants) with a live
        # question — grading is server-side and students can read this
        # payload in devtools.
        question_public = {k: v for k, v in (question or {}).items()
                           if k not in ("answer", "accept")}
        return {
            "session_id": session.session_id,
            "state": ps.phase,
            "question_index": session.question_index,
            "total_questions": session.total_questions,
            "current_question": question_public,
            "time_remaining": ps.time_remaining_seconds,
            "score": round(ps.current_score, 1),
            "max_score": round(ps.max_score, 1),
            "accuracy": round(
                ps.correct_count / max(1, ps.questions_answered) * 100, 1
            ),
            "stress_level": round(ps.stress_level, 2),
            "momentum": round(ps.momentum, 2),
            "consecutive_correct": ps.consecutive_correct,
            "consecutive_wrong": ps.consecutive_wrong,
            "was_correct": correct,
            "response_ms": response_ms,
            "coaching_interjection": coaching,
        }

    def _complete(self, session: SimulationSession, coaching: str = "",
                  correct: bool = False, response_ms: float = 0) -> dict:
        session.state = "complete"
        ps = session.controller.state
        total_q = session.total_questions
        answered = ps.questions_answered
        avg_ms = 0
        if answered > 0:
            qs = sim_store.get_answers(session.session_id)
            if qs:
                avg_ms = sum(q.get("response_ms", 0) for q in qs) / len(qs)

        resilience = round(
            1.0 - (ps.stress_level * 0.5) - (ps.consecutive_wrong * 0.05),
            2,
        )
        resilience = max(0, min(1, resilience))

        try:
            sim_store.complete_session(session.session_id, {
                "total_questions": total_q,
                "attempted": answered,
                "correct": ps.correct_count,
                "wrong": ps.wrong_count,
                "skipped": ps.skipped_count,
                "total_score": ps.current_score,
                "max_score": ps.max_score,
                "avg_response_ms": avg_ms,
                "stress_resilience": resilience,
            })
        except Exception:
            pass

        return {
            "session_id": session.session_id,
            "state": "complete",
            "score": round(ps.current_score, 1),
            "max_score": round(ps.max_score, 1),
            "accuracy": round(ps.correct_count / max(1, answered) * 100, 1),
            "questions_attempted": answered,
            "correct": ps.correct_count,
            "wrong": ps.wrong_count,
            "skipped": ps.skipped_count,
            "avg_response_ms": round(avg_ms, 0),
            "stress_resilience": resilience,
            "coaching_interjection": coaching,
        }

    def get_session(self, session_id: str) -> SimulationSession | None:
        return self._sessions.get(session_id)


engine = SimulationEngine()


# ── Confidence-driven helpers ───────────────────────────────────────────────


def _confidence_feedback(conf: int, correct: bool, topic: str) -> dict:
    """Generate metacognitive feedback based on confidence vs. correctness.

    Returns a dict with 'type' (overconfident|underconfident|well_calibrated)
    and 'text' for the frontend to display.
    """
    if conf < 0:
        return {}
    if correct and conf >= 80:
        return {"type": "well_calibrated", "text": f"Confident and correct on {topic}. You know this well."}
    if correct and conf < 40:
        return {"type": "underconfident", "text": f"You got {topic} right but weren't sure — trust your knowledge more here!"}
    if not correct and conf >= 70:
        return {"type": "overconfident", "text": f"Overconfidence on {topic} — you felt sure but missed. Let's review this together."}
    if not correct and conf < 30:
        return {"type": "well_calibrated", "text": f"Honest gap on {topic} — you knew you didn't know. That's the first step to learning."}
    return {"type": "well_calibrated", "text": ""}


def _apply_confidence_to_difficulty(controller, correct: bool, conf: int) -> None:
    """Adjust difficulty based on confidence × correctness interaction.

    - Correct + high confidence → push difficulty up (the learner has mastered this)
    - Correct + low confidence → hold difficulty (lucky guess, don't accelerate)
    - Wrong + high confidence → don't drop as hard (overconfidence needs re-exposure)
    - Wrong + low confidence → normal drop (honest gap, ease back in)
    """
    if conf < 0:
        return
    state = controller.state
    if correct and conf >= 80:
        # Accelerate: the learner knows this well
        state.current_difficulty = min(5.0, state.current_difficulty + 0.3)
    elif correct and conf < 40:
        # Hold: lucky guess, don't accelerate
        pass
    elif not correct and conf >= 70:
        # Overconfidence: don't bail them out — keep difficulty high so they
        # encounter the concept again at the same level
        state.current_difficulty = max(1.0, state.current_difficulty - 0.1)
    elif not correct and conf < 30:
        # Honest gap: normal difficulty drop
        state.current_difficulty = max(1.0, state.current_difficulty - 0.3)
