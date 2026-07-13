"""Pressure controller — modulates difficulty, time pressure, and traps based on student state."""

import math
import time
from dataclasses import dataclass, field
from enum import Enum


class Phase(Enum):
    WARMUP = "warmup"
    MAIN = "main"
    STRESS_SPIKE = "stress_spike"
    RECOVERY = "recovery"
    FINAL_SPRINT = "final_sprint"
    REVIEW = "review"


@dataclass
class PressureConfig:
    time_pressure: float = 0.5          # 0=untimed, 1=extreme (2/3 of real time)
    negative_marking: float = 0.25       # fraction of marks deducted for wrong
    distraction_density: float = 0.3     # 0=none, 1=every 2-3 problems
    recovery_enabled: bool = True        # coaching interjections
    fatigue_simulation: bool = True      # hard patch after 60% of session
    total_questions: int = 30
    time_limit_seconds: int = 3600       # default 60 min
    subject: str = "Phonetics"


@dataclass
class PressureState:
    phase: str = "warmup"
    current_difficulty: float = 1.0
    multiplier: float = 1.0
    consecutive_correct: int = 0
    consecutive_wrong: int = 0
    time_remaining_seconds: int = 0
    questions_answered: int = 0
    correct_count: int = 0
    wrong_count: int = 0
    skipped_count: int = 0
    current_score: float = 0.0
    max_score: float = 0.0
    last_coaching_at: float = 0.0
    momentum: float = 0.0  # -1 to 1, positive = improving
    stress_level: float = 0.0  # 0 to 1


class PressureController:
    def __init__(self, config: PressureConfig):
        self.config = config
        self.state = PressureState()
        self.question_start_time: float = 0
        self._stress_spike_triggered: bool = False
        self._fatigue_triggered: bool = False

    def start(self) -> PressureState:
        self.state.time_remaining_seconds = self.config.time_limit_seconds
        self.state.phase = "warmup"
        self.state.current_difficulty = 1.0
        return self.state

    def tick(self, elapsed_seconds: float) -> PressureState:
        self.state.time_remaining_seconds = max(
            0, self.state.time_remaining_seconds - int(elapsed_seconds)
        )
        time_pct = 1 - self.state.time_remaining_seconds / max(1, self.config.time_limit_seconds)
        question_pct = self.state.questions_answered / max(1, self.config.total_questions)

        # Phase transitions
        if question_pct < 0.1:
            self.state.phase = "warmup"
        elif question_pct < 0.7:
            self.state.phase = "main"
        elif question_pct < 0.85:
            self.state.phase = "stress_spike"
            self._stress_spike_triggered = True
        elif question_pct < 0.95:
            self.state.phase = "recovery" if not self._stress_spike_triggered else "recovery"
        else:
            self.state.phase = "final_sprint"

        # Fatigue: at 60-70% of session, inject difficulty bump
        if self.config.fatigue_simulation and 0.6 < question_pct < 0.75 and not self._fatigue_triggered:
            self._fatigue_triggered = True
            self.state.multiplier += 1.0

        # Adapt difficulty
        self._adapt_difficulty(elapsed_seconds)

        # Compute stress
        self.state.stress_level = min(1.0, (
            time_pct * self.config.time_pressure +
            (1 - self.state.time_remaining_seconds / max(1, self.config.time_limit_seconds)) * 0.5 +
            (self.state.consecutive_wrong * 0.15)
        ))

        return self.state

    def on_answer(self, was_correct: bool, fast: bool, response_ms: float) -> PressureState:
        max_time = self.config.time_limit_seconds / self.config.total_questions
        self.state.questions_answered += 1

        if was_correct:
            self.state.correct_count += 1
            self.state.consecutive_correct += 1
            self.state.consecutive_wrong = 0
            self.state.momentum = min(1.0, self.state.momentum + 0.15)
            points = 4 * self.state.current_difficulty
            self.state.current_score += points
        else:
            self.state.wrong_count += 1
            self.state.consecutive_wrong += 1
            self.state.consecutive_correct = 0
            self.state.momentum = max(-1.0, self.state.momentum - 0.25)
            penalty = 1 * self.config.negative_marking * self.state.current_difficulty
            self.state.current_score = max(0, self.state.current_score - penalty)

        self.state.max_score += 4 * self.state.current_difficulty

        # Stress spike: after 5 correct, inject harder question
        if self.state.consecutive_correct >= 5 and self.state.phase == "main":
            self.state.multiplier += 1.5
            self.state.consecutive_correct = 0

        return self.state

    def recover_from_wrong_streak(self) -> None:
        """Called after coaching has been delivered, if applicable."""
        if self.state.consecutive_wrong >= 3 and self.state.phase != "warmup":
            self.state.multiplier = max(0, self.state.multiplier - 1.0)
            self.state.consecutive_wrong = 0

    def on_skip(self) -> PressureState:
        self.state.skipped_count += 1
        self.state.questions_answered += 1
        return self.state

    def should_coach(self) -> bool:
        cooldown = time.time() - self.state.last_coaching_at > 15
        return cooldown and (
            self.state.consecutive_wrong >= 3 or
            self.state.momentum < -0.5 or
            self.state.stress_level > 0.7 or
            (self.state.time_remaining_seconds < 300 and self.state.questions_answered > 0)
        )

    def coach_triggered(self) -> None:
        self.state.last_coaching_at = time.time()

    def _adapt_difficulty(self, elapsed_seconds: float) -> None:
        d = self.state.current_difficulty * (1 + self.state.multiplier)

        if self.state.phase == "warmup":
            d = max(0.5, d * 0.5)
        elif self.state.phase == "stress_spike":
            d = max(1.5, d * 1.5)
        elif self.state.phase == "final_sprint":
            d = max(2.0, d * 1.3)

        self.state.current_difficulty = max(0.5, min(5.0, d))
