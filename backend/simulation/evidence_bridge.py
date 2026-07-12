"""Bridge: graded simulation answers → Evidence Object Graph → Cognitive Twin.

The simulation grades answers symbolically (engine._grade) but until now the
outcome never reached the twin — quiz performance was invisible to the
planner, the frontier and the learner's universe. This closes that gap.

Discipline: evidence must attach to a REAL node of the language graph
(evidence_graph docstring). Simulation questions carry a meta-linguistic
`topic` ("voicing", "morpheme", ...) that the word/root-oriented graph mostly
does not contain yet, so resolution is a ladder of exact normalized matches —
question topic first, then the session subject ("Morphology" → node
'morphology') — and emits NOTHING when neither resolves. No fuzzy matching:
mis-attributed evidence corrupts the belief state, silence does not. As the
graph gains concept nodes (node-curation task), more pools attach with zero
code change here.
"""

from __future__ import annotations

import re

_WORLD = None


def _get_world():
    """Share the cognitive loop's cached WorldModel; fall back to our own."""
    global _WORLD
    try:
        from cognitive_loop_routes import _get_world as loop_world
        return loop_world()
    except Exception:
        if _WORLD is None:
            from development_engine import WorldModel
            _WORLD = WorldModel()
        return _WORLD


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _norm_index(world) -> dict[str, str]:
    """normalized id/display → node_id, built once per WorldModel instance."""
    idx = getattr(world, "_evbridge_norm_index", None)
    if idx is None:
        idx = {}
        for nid, node in world.nodes.items():
            idx.setdefault(_norm(nid), nid)
            disp = node.get("display", "")
            if disp:
                idx.setdefault(_norm(disp), nid)
        world._evbridge_norm_index = idx
    return idx


def _variants(term: str) -> list[str]:
    spaced = term.replace("_", " ")
    out = [term, spaced, term.rstrip("s"), spaced.rstrip("s")]
    seen, uniq = set(), []
    for v in out:
        if v and v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def resolve_node(subject: str, topic: str, world=None) -> tuple[str, str] | None:
    """Return (node_id, resolved_from) or None. resolved_from ∈ {topic, subject}."""
    world = world or _get_world()
    idx = _norm_index(world)
    for source, term in (("topic", topic), ("subject", subject)):
        if not term:
            continue
        for cand in _variants(term):
            nid = idx.get(_norm(cand))
            if nid:
                return nid, source
    return None


def emit_quiz_evidence(
    user_id: int,
    subject: str,
    question: dict,
    was_correct: bool,
    response_ms: int = 0,
    session_id: str = "",
) -> dict | None:
    """Record one graded answer as quiz evidence and update the twin.

    Returns the belief summary (same shape as POST /loop/evidence) when the
    question resolved to a graph node, else None. Never raises past the
    caller's try/except — a twin hiccup must not fail the answer submission.
    """
    topic = question.get("topic", "")
    resolved = resolve_node(subject, topic)
    if resolved is None:
        return None
    node_id, resolved_from = resolved

    import cognitive_twin as twin
    from evidence_graph import EvidenceObject

    ev = EvidenceObject(
        student_id=f"u_{user_id}",
        node_id=node_id,
        source="quiz",
        outcome="correct" if was_correct else "incorrect",
        confidence=1.0,  # symbolic grading — the observation itself is certain
        meta={
            "subject": subject,
            "topic": topic,
            "difficulty": question.get("difficulty"),
            "session_id": session_id,
            "response_ms": response_ms,
            "resolved_from": resolved_from,
        },
    )
    belief = twin.update(ev)
    return {
        "node_id": belief.node_id,
        "mastery": round(belief.mastery, 4),
        "exposures": belief.exposures,
        "mastered": belief.mastered,
        "resolved_from": resolved_from,
    }
