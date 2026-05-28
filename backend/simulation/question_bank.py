"""Question bank — curated JEE problems with difficulty levels."""

import random
import hashlib

from .pressure import PressureState, Phase


class QuestionBank:
    def __init__(self, subject: str = "Physics"):
        self.subject = subject

    def get_question(self, state: PressureState, question_index: int = 0) -> dict:
        from . import _questions
        pool = getattr(_questions, f"POOL_{self.subject.upper()}", _questions.POOL_PHYSICS)

        # Difficulty-based filtering
        d = state.current_difficulty
        if state.phase == "warmup":
            d = max(0.5, d * 0.5)
        elif state.phase == "stress_spike":
            d = max(1.5, d * 1.5)

        # Find questions near the target difficulty
        close = [(abs(q.get("difficulty", 1) - d), q) for q in pool]
        close.sort(key=lambda x: x[0])

        # Pick from top 5 closest, with deterministic rotation by index
        candidates = [q for _, q in close[:5]]
        seed = hashlib.sha256(f"{question_index}:{state.phase}".encode()).digest()[0]
        selected = candidates[seed % len(candidates)]

        return {
            "query": selected["query"],
            "answer": selected["answer"],
            "topic": selected.get("topic", ""),
            "difficulty": round(selected.get("difficulty", 1.0), 1),
            "choices": selected.get("choices", []),
            "hint": selected.get("hint", ""),
        }
