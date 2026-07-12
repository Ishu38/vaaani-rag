"""Adaptive remediation generator — prescribes counter-strategies for cognitive errors."""

REMEDIATION_PATTERNS: dict[str, list[str]] = {
    "memorization_override": [
        "Instead of recalling the definition by rote, work through a fresh example word or sentence and label its parts before answering.",
        "Try explaining this concept to a peer without using the technical term — use only examples and everyday analogies.",
        "A definition is a compressed pattern. Rebuild it from three examples of your own, then compare with the textbook wording.",
    ],
    "conceptual_gap": [
        "Go back one level deeper — the concept this depends on is missing. Identify it and review from first principles.",
        "Watch a video or read a visual explanation of the underlying principle. Do not move forward until you can explain it in your own words.",
        "Solve 3 simpler problems on this exact sub-topic before attempting this difficulty again.",
    ],
    "terminology_confusion": [
        "You mixed up two technical terms (like phoneme vs morpheme, or synonym vs hyponym). Make a two-column card: term, definition, one example each — and quiz yourself both directions.",
        "Slow down on the naming step. First describe what you see in plain words, THEN attach the technical label.",
        "Keep a 'confusable pairs' log — note which two terms you swapped, review it weekly.",
    ],
    "spelling_sound_conflation": [
        "You counted letters, not sounds. Say the word aloud slowly and tap once per sound — spelling is a costume, sound is the body.",
        "Practice with tricky words where letters and sounds diverge: 'box' (4 sounds), 'thick' (3 sounds), 'knee' (2 sounds).",
        "Before answering any sound question, close your eyes and listen to the word — don't look at its spelling.",
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
        "Close the book and draw the structure from scratch — the syntax tree, the morpheme breakdown, or the vowel chart. Only then start answering.",
        "Use the Feynman explain-it-back mode to describe the structure in words. If you can't draw it, you don't see it.",
        "For sentence-structure problems, practice bracketing the sentence into chunks on paper before labelling anything.",
    ],
    "l1_transfer": [
        "This is your mother tongue's pattern showing through — a normal stage, not a mistake in thinking. Name the two rules side by side: how your language does it, and how English does it.",
        "Collect three more examples of this exact transfer pattern. Once you can predict where it appears, you can catch it before it happens.",
    ],
    "overgeneralisation": [
        "You applied a real rule beyond its limits (like 'goed' for 'went'). Good news: you clearly know the rule. Now list its exceptions on one card and drill just those.",
        "For the next 10 items, ask before answering: 'Is this one of the regulars, or one of the rebels?' Irregulars have to be memorised as a family.",
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
