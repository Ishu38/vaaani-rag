"""Real-time coaching interjections — contextual psychological prompts during simulation."""

from .pressure import PressureState, Phase


COACH_SCRIPTS = {
    "impulsive_loop": [
        "You're in an impulsive loop. Pause 5 seconds. What changed from the question before this?",
        "Too fast. Read the question aloud in your head once before touching any numbers.",
        "Impulsive mode detected. Deliberately slow down — accuracy > speed right now.",
    ],
    "tunneling": [
        "You're tunneling on one question. Flag it and move on. This is costing you easier ones.",
        "Stop. Skip this one. Your time-per-question is too high — bank easier marks first.",
        "Flagged. Next. You can come back to this if time permits.",
    ],
    "high_speed_low_accuracy": [
        "You're answering fast but wrong — {acc_pct}% accuracy vs your normal. Add 15 seconds per question.",
        "Slow is smooth, smooth is fast. You're losing more marks to speed-errors than you're gaining from speed.",
    ],
    "low_speed_good_accuracy": [
        "Your accuracy is high but you're behind schedule. Trust your first instinct more.",
        "Great accuracy — but you need to pick up pace. Try to cut 20 seconds per question without dropping quality.",
    ],
    "final_minutes": [
        "5 minutes remain. Review flagged questions first. Don't start new problems.",
        "Last moments. Check your flagged questions. Trust your work on the rest.",
    ],
    "stress_collapse": [
        "Breathe. You know this material. The pressure is making you rush. Reset — count to 5.",
        "Your cognitive load is high. Take a 10-second mental reset. Close your eyes. Then continue.",
    ],
    "momentum_loss": [
        "That's 3 wrong in a row. Don't spiral. The next question is a new chance. Reset mentally.",
        "Streak broken. Doesn't matter. What matters is the next question. Let the past ones go.",
    ],
    "negative_marking_trap": [
        "This question has negative marking. If you're less than 70% sure, SKIP it. A blank is better than a wrong.",
        "Trap alert. This looks like a standard question but the wording is tweaked. Read every word carefully.",
    ],
}


class CoachInterjector:
    def __init__(self):
        self._interjection_count: int = 0
        self._last_message: str = ""

    def generate(
        self,
        state: PressureState,
        was_correct: bool | None = None,
        response_ms: float = 0,
    ) -> str:
        import random

        candidates: list[str] = []
        acc = state.correct_count / max(1, state.questions_answered) * 100

        # Priority 1: consecutive wrong → momentum loss / impulsive
        if state.consecutive_wrong >= 3:
            if response_ms < 12000:
                candidates.extend(COACH_SCRIPTS["impulsive_loop"])
            else:
                candidates.extend(COACH_SCRIPTS["momentum_loss"])

        # Priority 2: stress level
        if state.stress_level > 0.8:
            candidates.extend(COACH_SCRIPTS["stress_collapse"])

        # Priority 3: speed-accuracy tradeoff
        if response_ms < 10000 and acc < 50:
            candidates.extend(COACH_SCRIPTS["high_speed_low_accuracy"])
        elif response_ms > 60000 and acc > 70:
            candidates.extend(COACH_SCRIPTS["low_speed_good_accuracy"])

        # Priority 4: time pressure
        if state.time_remaining_seconds < 300:
            candidates.extend(COACH_SCRIPTS["final_minutes"])

        # Priority 5: negative marking
        if state.correct_count < state.questions_answered * 0.4 and state.consecutive_wrong >= 2:
            candidates.extend(COACH_SCRIPTS["negative_marking_trap"])

        if not candidates:
            return ""

        message = random.choice(candidates)
        if "{acc_pct}" in message:
            message = message.format(acc_pct=f"{acc:.0f}%")

        # Don't repeat the same message twice
        if message == self._last_message and len(candidates) > 1:
            candidates.remove(message)
            message = random.choice(candidates)

        self._last_message = message
        return message
