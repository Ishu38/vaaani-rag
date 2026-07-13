"""Unified Learner State — single query for the full learner model.

Architecture position: [Linguistic World Model | Learner State] box.

Queries the cognitive twin (BKT), credal beliefs (Beta), cognitive
fingerprint, expedition state, and evidence history in one call.
Returns a structured dict the whole pipeline can consume.
No LLM, no SLM — purely symbolic aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class LearnerState:
    student_id: str

    # Twin (BKT mastery) summary
    nodes_tracked: int = 0
    nodes_mastered: int = 0
    avg_mastery: float = 0.0
    mastered_nodes: list = field(default_factory=list)

    # Credal (Beta uncertainty) summary
    credal_nodes: int = 0
    avg_uncertainty: float = 0.0

    # Expedition
    active_expedition: dict | None = None

    # Cognitive fingerprint
    strengths: list = field(default_factory=list)
    weaknesses: list = field(default_factory=list)
    primary_weakness: str = ""
    accuracy: float = 0.0
    total_analyzed: int = 0

    # Spaced review
    review_due_count: int = 0

    # Evidence count
    total_evidence: int = 0

    # Cold start flag
    is_cold_start: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_ready_to_plan(self) -> bool:
        """Has enough evidence for the planner to make informed decisions."""
        return not self.is_cold_start and self.nodes_tracked >= 3

    @property
    def zpd_summary(self) -> str:
        """Human-readable summary of the learner's readiness for growth."""
        if self.is_cold_start:
            return "Just starting — not enough data yet."
        mastered = ", ".join(self.mastered_nodes[:5]) or "nothing specific yet"
        weak = self.primary_weakness or "no clear weakness yet"
        return (f"Tracking {self.nodes_tracked} concepts, "
                f"{self.nodes_mastered} mastered. "
                f"Top skills: {mastered}. "
                f"Weakest area: {weak}. "
                f"Accuracy: {self.accuracy:.0%}.")


def get_learner_state(student_id: str) -> LearnerState:
    """Query all learner model subsystems and return a unified state."""
    import cognitive_twin as twin
    import credal
    from evidence_graph import count as evidence_count

    state = LearnerState(student_id=str(student_id))

    # ── Cognitive Twin ──────────────────────────────────────────
    try:
        snap = twin.snapshot(student_id)
        state.nodes_tracked = len(snap)
        state.nodes_mastered = sum(1 for b in snap.values() if b.mastered)
        state.avg_mastery = (sum(b.mastery for b in snap.values()) /
                             len(snap)) if snap else 0.0
        state.mastered_nodes = sorted(
            [n for n, b in snap.items() if b.mastered],
            key=lambda n: -snap[n].mastery)[:10]
        if snap:
            state.is_cold_start = False
    except Exception:
        pass

    # ── Credal (Beta uncertainty) ───────────────────────────────
    try:
        csnap = credal.snapshot(student_id)
        state.credal_nodes = len(csnap)
        state.avg_uncertainty = (sum(c.sd for c in csnap.values()) /
                                  len(csnap)) if csnap else 0.0
    except Exception:
        pass

    # ── Expedition ──────────────────────────────────────────────
    try:
        import expedition as exp_mod
        exp = exp_mod.get_active(student_id)
        if exp:
            state.active_expedition = {
                "anchor": exp.anchor_id,
                "display": exp.display,
                "tier": exp.tier,
                "step": exp.steps_done + 1,
                "total": exp.steps_total,
                "status": exp.status,
            }
    except Exception:
        pass

    # ── Cognitive Fingerprint ───────────────────────────────────
    try:
        from cognitive.fingerprint import build_fingerprint
        fp = build_fingerprint(int(student_id))
        state.strengths = fp.get("strengths", []) or []
        state.weaknesses = fp.get("weaknesses", []) or []
        s = fp.get("summary", {}) or {}
        state.accuracy = s.get("accuracy", 0) / 100.0 if s.get("accuracy") else 0.0
        state.total_analyzed = s.get("total_analyzed", 0) or 0
        pw = s.get("primary_weakness_label", "")
        state.primary_weakness = pw if pw not in ("None", "No data yet", "Unknown", "") else ""
    except Exception:
        pass

    # ── Spaced review ───────────────────────────────────────────
    try:
        from adaptive.service import due_for_review
        due = due_for_review(student_id)
        state.review_due_count = len(due) if due else 0
    except Exception:
        pass

    # ── Evidence count ──────────────────────────────────────────
    try:
        state.total_evidence = evidence_count(student_id)
    except Exception:
        pass

    return state
