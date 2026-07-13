"""Vaaani Discovery Orchestrator — deterministic, no LLM/SLM.

Pipeline: PERCEPTION → Evidence Object Graph → Neuro-Symbolic Reasoner →
[Linguistic World Model | Learner State] → Linguistic Development Engine →
Pedagogical Decision Engine → Evidence-driven Learning Action

The orchestrator is the master control: it receives a discovery context
(grade, mastered sounds, source page, L1), locates the learner in the
cognitive twin, and drives the deterministic pedagogical pipeline:

    development_engine.frontier()       → ZPD frontier
    pedagogical_planner.select_activity() → best mission (decision theory)
    activity_generator.generate()         → learner-facing activity

Every mission carries credits (node ids) so evidence closes the loop.
No LLM, no SLM — every decision is explainable and auditable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import COMMUNITIES_PATH, DATA_DIR, GRAPH_PATH


# ── Learner Context Builder ──────────────────────────────────────────

def build_learner_context(
    user: dict | None,
    discovery_context: dict | None = None,
) -> dict:
    """Build a structured learner profile from the discovery context.

    Returns a dict with keys the pedagogical planner can consume:
      student_id, grade, l1, mastered_sounds, mastered_families,
      weak_patterns, recent_errors, current_sound, source_page
    """
    ctx = discovery_context or {}
    out: dict[str, Any] = {}

    if user:
        out["student_id"] = str(user.get("id", ""))
        out["name"] = user.get("name") or user.get("display_name") or ""

    out["source"] = ctx.get("source", "")

    grade = ctx.get("grade")
    if grade is not None:
        try:
            out["grade"] = int(grade)
        except (TypeError, ValueError):
            pass

    out["l1"] = ctx.get("l1") or ctx.get("native_language") or ""
    out["current_sound"] = ctx.get("sound", "")
    out["current_topic"] = ctx.get("topic", "")

    completed = ctx.get("completed", [])
    if isinstance(completed, str):
        completed = [s.strip() for s in completed.split(",") if s.strip()]
    out["completed"] = completed

    mastered = ctx.get("mastered_sounds", [])
    if isinstance(mastered, str):
        mastered = [s.strip() for s in mastered.split(",") if s.strip()]
    out["mastered_sounds"] = mastered

    families = ctx.get("unlocked_word_families", [])
    if isinstance(families, str):
        families = [s.strip() for s in families.split(",") if s.strip()]
    out["unlocked_word_families"] = families

    roots = ctx.get("unlocked_roots", [])
    if isinstance(roots, str):
        roots = [s.strip() for s in roots.split(",") if s.strip()]
    out["unlocked_roots"] = roots

    weak = ctx.get("weak_patterns") or ctx.get("current_weak_areas") or []
    if isinstance(weak, str):
        weak = [s.strip() for s in weak.split(",") if s.strip()]
    out["weak_patterns"] = weak

    confused = ctx.get("recent_errors") or ctx.get("recently_confused_concepts") or []
    if isinstance(confused, str):
        confused = [s.strip() for s in confused.split(",") if s.strip()]
    out["recent_errors"] = confused

    missions = ctx.get("completed_missions")
    if missions is not None:
        try:
            out["completed_missions"] = int(missions)
        except (TypeError, ValueError):
            pass

    return out


# ── Deterministic Discovery Engine ───────────────────────────────────

def run_discovery_orchestrator(
    student_id: str,
    discovery_context: dict | None = None,
) -> dict:
    """The deterministic discovery pipeline — no LLM anywhere.

    Args:
        student_id: the learner's id (maps to cognitive twin)
        discovery_context: optional hints (grade, mastered sounds, L1)

    Returns:
        dict with {mission, decision, activity, companion, deep_link?}
        ready to return to the frontend.
    """
    import cognitive_twin as twin
    import expedition as exp_mod
    from activity_generator import generate
    from companion import companion_block
    from development_engine import WorldModel
    from pedagogical_planner import select_activity

    world = WorldModel()
    ctx = discovery_context or {}

    # ── Cold-start seeding ──────────────────────────────────────
    # If the twin has no data for this student, seed it with the
    # declared mastered concepts from the discovery_context so the
    # planner has something to work from — each declared-mastered
    # node gets an initial-strong belief.
    snap = twin.snapshot(student_id)
    if not snap and ctx:
        _seed_twin_from_context(student_id, ctx, world)

    # ── Run the deterministic pipeline ──────────────────────────
    decision = select_activity(student_id, world)
    activity = generate(decision, world)
    exp = exp_mod.get_active(student_id)
    companion = companion_block(student_id, exp)

    resp: dict = {
        "decision": {
            "kind": decision.kind,
            "node_id": decision.node_id,
            "display": decision.display,
            "p_success": round(decision.p_success, 3),
            "reason": decision.reason,
            "edge_key": decision.edge_key,
        },
        "activity": activity.to_dict(),
        "companion": companion,
    }

    if exp is not None:
        resp["expedition"] = {
            "anchor": exp.anchor_id,
            "display": exp.display,
            "tier": exp.tier,
            "step": exp.steps_done + 1,
            "total": exp.steps_total,
            "status": exp.status,
        }

    # ── Deep-link hint ──────────────────────────────────────────
    tier = activity.activity_type
    if tier in ("sound_detective",):
        resp["deep_link"] = f"/sound-lab?s={ctx.get('current_sound', '')}"
        resp["deep_link_label"] = "Open Sound Lab"
    elif tier == "bridge_match" and ctx.get("l1"):
        resp["deep_link"] = "/language-map"
        resp["deep_link_label"] = "Open Language Map"

    # ── Available discovery paths (for the frontend sidebar) ────
    touched = list(ctx.get("mastered_sounds", []))
    if not touched:
        touched = ctx.get("completed", [])
    paths = _available_discovery_paths(touched, ctx.get("sound", ""),
                                        grade=ctx.get("grade"))
    if paths:
        resp["discovery_paths"] = paths

    return resp


def _seed_twin_from_context(student_id: str, ctx: dict,
                             world: 'WorldModel') -> None:
    """Cold-start: seed the cognitive twin from discovery context.

    Each declared-mastered concept gets a strong initial node belief.
    CASCADE: also seeds edge beliefs from L1 graft when L1 is provided.
    """
    import cognitive_twin as twin
    from evidence_graph import EvidenceObject

    sounds = ctx.get("mastered_sounds", [])
    families = ctx.get("unlocked_word_families", [])
    roots = ctx.get("unlocked_roots", [])
    declared = [s.lower().strip() for s in sounds + families + roots
                if s and s.strip()]

    for node_id in world.nodes:
        disp = world.display(node_id).lower()
        if any(d in disp or disp in d for d in declared):
            try:
                ev = EvidenceObject(
                    student_id, node_id, "mission", "correct",
                    confidence=0.85,
                    meta={"seed": True, "reason": "declared_mastered"},
                )
                twin.update(ev)
            except Exception:
                pass

    # CASCADE: seed edge beliefs from L1 graft
    l1 = ctx.get("l1", "")
    if l1:
        try:
            from development_engine import get_world_edges
            from l1_graft import seed_edges_from_l1
            edges = get_world_edges()
            seeded = seed_edges_from_l1(
                student_id, edges, world.display, l1)
            if seeded:
                pass  # edge beliefs seeded from L1
        except Exception:
            pass


# ── Available Discovery Paths ────────────────────────────────────────

def _available_discovery_paths(
    completed_sounds: list[str],
    current_sound: str,
    grade: int | None = None,
) -> list[str]:
    """Scan the knowledge graph for word-family communities the learner
    hasn't yet explored, returning short descriptions usable as mission seeds."""
    paths: list[str] = []
    try:
        if COMMUNITIES_PATH.exists():
            with open(COMMUNITIES_PATH) as f:
                communities = json.load(f)

            touched = {s.lower().strip() for s in completed_sounds if s.strip()}
            if current_sound:
                touched.add(current_sound.lower().strip())

            for c in communities:
                title = c.get("title", "")
                nodes = c.get("nodes", [])
                if not title or not nodes:
                    continue
                overlap = any(
                    n.lower() in touched or any(t in n.lower() for t in touched)
                    for n in nodes
                )
                if overlap:
                    new_nodes = [n for n in nodes if n.lower() not in touched]
                    if new_nodes:
                        paths.append(
                            f"Word family: {title} — "
                            f"you know {len(nodes) - len(new_nodes)}/{len(nodes)} words; "
                            f"still to discover: {', '.join(new_nodes[:4])}"
                        )

            if not paths and communities:
                top = communities[:3]
                for c in top:
                    title = c.get("title", "")
                    nodes = c.get("nodes", [])[:3]
                    if title and nodes:
                        paths.append(
                            f"New territory — word family: {title} "
                            f"(e.g., {', '.join(nodes)})"
                        )

    except Exception:
        pass

    if grade is not None:
        try:
            from graph_curriculum import words_for_grade
            curr = "ncert"
            required_words = words_for_grade(curr, int(grade))
            touched_words = {s.lower().strip() for s in completed_sounds
                           if s.strip()}
            remaining_words = [w for w in required_words
                             if w.lower().strip() not in touched_words][:8]
            if remaining_words:
                paths.append(
                    f"CURRICULUM (NCERT Grade {grade}): "
                    f"still to master — {', '.join(remaining_words[:6])}"
                )
        except Exception:
            pass

    return paths[:8]
