"""Cognitive fingerprint builder — aggregates error patterns into a student profile."""

from collections import defaultdict
from dataclasses import dataclass, field
from .store import store
from .classifier import ERROR_TYPES


ERROR_LABELS = {
    "memorization_override": "Rote Memorization",
    "conceptual_gap": "Conceptual Gap",
    "terminology_confusion": "Term Mix-ups",
    "spelling_sound_conflation": "Letters vs Sounds",
    "l1_transfer": "Mother-Tongue Transfer",
    "overgeneralisation": "Rule Overgeneralisation",
    "overconfidence": "Overconfidence",
    "underconfidence": "Underconfidence",
    "impulsive": "Impulsive Solving",
    "shortcut_dependency": "Shortcut Dependency",
    "fragile_understanding": "Fragile Understanding",
    "visualization_weakness": "Visualization Gap",
    "no_error": "Correct",
}


@dataclass
class Fingerprint:
    user_id: int
    error_breakdown: dict[str, int] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    biases: dict[str, str] = field(default_factory=dict)
    topics: dict[str, dict] = field(default_factory=dict)
    resilience_score: float = 0.5
    summary: str = ""


def build_fingerprint(user_id: int) -> dict:
    events = store.get_recent_events(user_id, limit=200)
    calib = store.confidence_calibration(user_id)

    if not events:
        return _empty_fingerprint(user_id)

    error_counts: dict[str, int] = defaultdict(int)
    topic_errors: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    correct_count = 0
    total_count = len(events)

    for e in events:
        if e.get("error_type", "") == "no_error" or e.get("actual_correct"):
            correct_count += 1
        else:
            et = e.get("error_type", "conceptual_gap")
            error_counts[et] += 1
            topic = e.get("topic", "general")
            topic_errors[topic][et] += 1

    error_count = total_count - correct_count
    accuracy = correct_count / max(1, total_count) * 100

    # Strengths = topics with highest accuracy
    topic_acc: dict[str, dict] = {}
    for topic, errs in topic_errors.items():
        topic_total = sum(errs.values())
        topic_correct = topic_total  # approximate, we don't track per-topic total
        topic_acc[topic] = {
            "error_count": topic_total,
            "top_error": max(errs, key=errs.get) if errs else "none",
        }

    strengths = sorted(
        [t for t, d in topic_acc.items() if d["error_count"] < 2],
        key=lambda t: topic_acc[t]["error_count"],
    )[:5]
    weaknesses = sorted(
        [t for t, d in topic_acc.items() if d["error_count"] >= 3],
        key=lambda t: -topic_acc[t]["error_count"],
    )[:5]

    # Cognitive bias profile
    biases = {}
    if error_counts.get("overconfidence", 0) > error_counts.get("underconfidence", 0) * 3:
        biases["tendency"] = "overconfidence_dominant"
        biases["description"] = "Consistently overestimates ability — needs calibration training"
    elif error_counts.get("underconfidence", 0) > error_counts.get("overconfidence", 0) * 3:
        biases["tendency"] = "underconfidence_dominant"
        biases["description"] = "Undersells own knowledge — needs confidence building"

    if error_counts.get("impulsive", 0) > max(1, total_count) * 0.3:
        biases["tendency"] = biases.get("tendency", "") + "_impulsive"
        biases["speed_issue"] = "Answers too quickly without verification"

    if error_counts.get("shortcut_dependency", 0) > max(1, total_count) * 0.2:
        biases["shortcut_issue"] = "Relies on pattern matching over understanding"

    # Resilience score — the metric the UI promises: of the answers given
    # immediately after a miss, what fraction were correct? (Bounce-back
    # rate, not accuracy.) Events arrive newest-first; walk them oldest-first.
    def _ok(e: dict) -> bool:
        return e.get("error_type", "") == "no_error" or bool(e.get("actual_correct"))

    chrono = list(reversed(events))
    followups = [
        _ok(nxt) for prev, nxt in zip(chrono, chrono[1:]) if not _ok(prev)
    ]
    # No miss yet (or nothing after the only miss): resilience is untested.
    resilience = (sum(followups) / len(followups)) if followups else None

    return {
        "user_id": user_id,
        "summary": {
            "total_analyzed": total_count,
            "accuracy": round(accuracy, 1),
            "primary_weakness": max(error_counts, key=error_counts.get) if error_counts else "none",
            "primary_weakness_label": ERROR_LABELS.get(
                max(error_counts, key=error_counts.get), "Unknown"
            ) if error_counts else "None",
            "error_frequency": error_count,
            "dominant_error_ratio": round(
                max(error_counts.values()) / max(1, error_count) * 100, 1
            ) if error_counts else 0,
        },
        "error_breakdown": {
            ERROR_LABELS.get(k, k): v for k, v in sorted(
                error_counts.items(), key=lambda x: -x[1]
            )
        },
        "strengths": strengths,
        "weaknesses": weaknesses,
        "biases": biases,
        "topics": topic_acc,
        "resilience_score": round(resilience, 2) if resilience is not None else None,
        "resilience_n": len(followups),
        "confidence_calibration": calib,
    }


def _empty_fingerprint(user_id: int) -> dict:
    return {
        "user_id": user_id,
        "summary": {"total_analyzed": 0, "accuracy": 0, "primary_weakness": "none",
                     "primary_weakness_label": "No data yet", "error_frequency": 0,
                     "dominant_error_ratio": 0},
        "error_breakdown": {},
        "strengths": [],
        "weaknesses": [],
        "biases": {},
        "topics": {},
        "resilience_score": None,
        "resilience_n": 0,
        "confidence_calibration": {},
    }
