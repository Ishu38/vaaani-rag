"""Adaptive remediation generator — prescribes counter-strategies for cognitive errors."""

REMEDIATION_PATTERNS: dict[str, list[str]] = {
    "memorization_override": [
        "Instead of recalling the formula by rote, draw a diagram of the situation and label forces/quantities before applying any equation.",
        "Try explaining this concept to a peer without using any formula — use only diagrams and real-world analogies.",
        "The formula is a compressed representation of a geometric truth. Revisit the derivation step by step.",
    ],
    "conceptual_gap": [
        "Go back one level deeper — the concept this depends on is missing. Identify it and review from first principles.",
        "Watch a video or read a visual explanation of the underlying principle. Do not move forward until you can explain it in your own words.",
        "Solve 3 simpler problems on this exact sub-topic before attempting this difficulty again.",
    ],
    "algebraic_slip": [
        "After solving, substitute your answer back into the original equation to verify. This habit catches 80% of algebraic errors.",
        "Slow down on the simplification step. Write intermediate lines — don't skip steps in your head.",
        "Keep a 'slip log' — note what kind of algebraic error you made, review it weekly.",
    ],
    "dimensional_error": [
        "Before plugging numbers, write the units for each term. Cancel units across the equation. The result's unit must match what you're solving for.",
        "Practice dimensional analysis problems for 20 minutes. This is a skill, not a disability.",
        "When your answer seems 'too big' or 'too small', use order-of-magnitude estimation to sanity-check.",
    ],
    "overconfidence": [
        "Rate your confidence before seeing the answer. Track it. If you're consistently overconfident, you're skipping the verification step.",
        "For the next 10 problems, deliberately state one reason why your answer MIGHT be wrong before submitting.",
        "Your gut is fast but not yet accurate. Force a 30-second deliberate review before every submission.",
    ],
    "underconfidence": [
        "You're right more often than you think. Track your accuracy at each confidence level — your 3/5-rated answers are probably correct.",
        "Trust your first instinct on problems where you've practiced the concept. Doubt only when you haven't seen the pattern before.",
        "Start the next problem with the assumption you can solve it. Write the first step before evaluating difficulty.",
    ],
    "impulsive": [
        "Use the 3-second rule: read the question, pause 3 seconds, re-read it, then start solving.",
        "Impulsive solving costs you 30% accuracy. Deliberately set a minimum solve time of 60 seconds per question.",
        "Before submitting, ask: 'What did the question actually ask?' vs 'What did I assume it asked?'",
    ],
    "shortcut_dependency": [
        "The shortcut works for a narrow pattern. You applied it to the wrong pattern. Practice identifying WHEN a formula applies vs when it doesn't.",
        "For the next session: solve each problem twice — once with the shortcut, once with the full method. Confirm they match.",
        "This is a sign of intelligent pattern matching gone wrong. The fix: always pause and check if the problem's structure matches the shortcut's requirements.",
    ],
    "fragile_understanding": [
        "You understood it once but can't reproduce it. Spaced repetition needed — revisit this topic every 3 days for 2 weeks.",
        "Teach this concept to a classmate. The gaps in your understanding will reveal themselves when you try to explain it.",
        "Solve variations of this problem: change one parameter, change the framing, add a twist. Build robustness.",
    ],
    "visualization_weakness": [
        "Close the book and draw the free-body diagram / circuit / geometric figure from scratch. Only then start solving.",
        "Use the Feynman explain-it-back mode to describe the visual setup in words. If you can't draw it, you don't see it.",
        "For 3D geometry and vector problems, practice with physical objects or interactive 3D visualizers before solving on paper.",
    ],
    "unit_confusion": [
        "Convert ALL units to SI before starting. Write them next to each number. Most errors come from mixing cm with m or g with kg.",
        "Create a unit conversion reference chart and tape it to your desk. Use it deliberately for 2 weeks until it becomes automatic.",
    ],
    "sign_error": [
        "Your method is right — the sign flipped. Check: subtraction order, vector direction conventions, and whether you distributed the negative.",
        "When rewriting equations from one line to the next, mentally check each term's sign. This simple habit catches 90% of sign errors.",
    ],
}

GENERIC_REMEDIATION = (
    "Review the underlying concept from first principles. "
    "Re-attempt the problem after 24 hours. If still struggling, flag for teacher review."
)


def generate_remediation(error_type: str, topic: str = "") -> str:
    options = REMEDIATION_PATTERNS.get(error_type, [GENERIC_REMEDIATION])
    import hashlib
    seed = hashlib.sha256(f"{error_type}:{topic}".encode()).digest()[0]
    return options[seed % len(options)]
