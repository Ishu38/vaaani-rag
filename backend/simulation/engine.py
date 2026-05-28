"""Simulation engine — state machine that drives exam pressure simulation."""

import uuid
import time
from dataclasses import dataclass, field
from enum import Enum

from .pressure import PressureController, PressureConfig, PressureState, Phase
from .coach import CoachInterjector
from .question_bank import QuestionBank
from .store import sim_store


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
            subject=config.get("subject", "Physics"),
        )
        session_id = uuid.uuid4().hex[:16]
        controller = PressureController(pc)
        controller.start()
        qb = QuestionBank()

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

        return self._session_response(session, q)

    def answer(self, session_id: str, answer: str, confidence_1to5: int = 0) -> dict:
        session = self._sessions.get(session_id)
        if not session:
            return {"error": "session not found", "state": "unknown"}
        if session.state == "complete":
            return {"error": "session completed", "state": "complete"}

        q = session.current_question
        if not q:
            return {"error": "no active question"}

        response_ms = (time.time() - session.controller.question_start_time) * 1000
        correct = answer.strip().lower() == q.get("answer", "").strip().lower()

        session.controller.on_answer(correct, response_ms < 15000, response_ms)
        session.controller.tick(time.time() - session.started_at - session.controller.state.time_remaining_seconds * 0)

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

        # Next question
        session.question_index += 1
        if session.question_index >= session.config.total_questions or (
            session.controller.state.time_remaining_seconds <= 0
        ):
            return self._complete(session, coaching, correct, response_ms)

        next_q = session.question_bank.get_question(
            session.controller.state,
            question_index=session.question_index,
        )
        session.current_question = next_q

        return self._session_response(session, next_q, coaching, correct, response_ms)

    def skip(self, session_id: str) -> dict:
        session = self._sessions.get(session_id)
        if not session:
            return {"error": "session not found"}
        session.controller.on_skip()
        session.question_index += 1
        if session.question_index >= session.config.total_questions:
            return self._complete(session, "", False, 0)
        next_q = session.question_bank.get_question(
            session.controller.state, question_index=session.question_index
        )
        session.current_question = next_q
        return self._session_response(session, next_q)

    def _session_response(
        self, session: SimulationSession, question: dict,
        coaching: str = "", correct: bool | None = None, response_ms: float = 0
    ) -> dict:
        ps = session.controller.state
        return {
            "session_id": session.session_id,
            "state": ps.phase,
            "question_index": session.question_index,
            "total_questions": session.total_questions,
            "current_question": question,
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
