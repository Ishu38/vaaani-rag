"""FastAPI router for the cognitive loop — /loop/* endpoints.

The whole architecture over HTTP, no LLM/SLM anywhere in the path:

  GET  /loop/mission/{student_id}      planner decision + generated activity
  GET  /loop/discover/{student_id}     discovery orchestrator (seeded from context)
  POST /loop/evidence                  learner outcome -> twin update
  GET  /loop/frontier/{student_id}     Development Engine's ZPD frontier
  GET  /loop/twin/{student_id}         belief-state summary
  GET  /loop/calibration/{student_id}  Metacognitive Evaluation table

v0 note: student_id is taken from the path (same trust level as existing
/learning endpoints' session flow should be added when merging with auth).
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

import cognitive_twin as twin
from activity_generator import generate
from development_engine import WorldModel, frontier
from evidence_graph import EvidenceObject, OUTCOMES, SOURCES
from pedagogical_planner import select_activity

router = APIRouter(prefix="/loop", tags=["cognitive-loop"])

_world: WorldModel | None = None


def _get_world() -> WorldModel:
    global _world
    if _world is None:
        _world = WorldModel()
    return _world


class EvidenceIn(BaseModel):
    student_id: str
    node_id: str
    edge_key: str = ""       # CASCADE: edge-level evidence
    source: str = Field(description=f"one of {SOURCES}")
    outcome: str = Field(description=f"one of {OUTCOMES}")
    confidence: float = 1.0
    meta: dict = Field(default_factory=dict)


@router.get("/mission/{student_id}")
def get_mission(student_id: str, l1: str = Query("")) -> dict:
    import expedition as exp_mod
    from companion import companion_block

    world = _get_world()

    # CASCADE: seed L1 edges on cold start
    if l1 and l1 in ("hi", "bn", "ta", "te", "gu", "pa", "ml", "kn", "or"):
        try:
            from l1_graft import seed_edges_from_l1
            from development_engine import get_world_edges
            snap = twin.snapshot_edges(student_id)
            if not snap:
                seed_edges_from_l1(student_id, get_world_edges(),
                                   world.display, l1)
        except Exception:
            pass

    decision = select_activity(student_id, world)
    activity = generate(decision, world)
    exp = exp_mod.get_active(student_id)
    resp = {"decision": {
                "kind": decision.kind, "node_id": decision.node_id,
                "display": decision.display,
                "p_success": round(decision.p_success, 3),
                "reason": decision.reason,
                "edge_key": decision.edge_key},
            "activity": activity.to_dict(),
            "companion": companion_block(student_id, exp)}
    if exp is not None:
        resp["expedition"] = {"anchor": exp.anchor_id, "display": exp.display,
                              "tier": exp.tier, "step": exp.steps_done + 1,
                              "total": exp.steps_total, "status": exp.status}
    return resp


@router.get("/discover/{student_id}")
def get_discovery(
    student_id: str,
    grade: int | None = Query(None, ge=1, le=12),
    sound: str = Query(""),
    l1: str = Query(""),
    source: str = Query(""),
    mastered_sounds: str = Query(""),
    unlocked_families: str = Query(""),
    unlocked_roots: str = Query(""),
    completed_missions: int | None = Query(None),
    weak_patterns: str = Query(""),
) -> dict:
    """Discovery orchestrator — deterministic, no LLM.

    Accepts discovery context from the calling page (IPA, Sound Lab, etc.)
    and runs the full pedagogical pipeline to return a mission."""
    ctx: dict = {}
    if grade:
        ctx["grade"] = grade
    if sound:
        ctx["sound"] = sound
    if l1:
        ctx["l1"] = l1
    if source:
        ctx["source"] = source
    if mastered_sounds:
        ctx["mastered_sounds"] = [s.strip() for s in mastered_sounds.split(",") if s.strip()]
    if unlocked_families:
        ctx["unlocked_word_families"] = [s.strip() for s in unlocked_families.split(",") if s.strip()]
    if unlocked_roots:
        ctx["unlocked_roots"] = [s.strip() for s in unlocked_roots.split(",") if s.strip()]
    if completed_missions is not None:
        ctx["completed_missions"] = completed_missions
    if weak_patterns:
        ctx["weak_patterns"] = [s.strip() for s in weak_patterns.split(",") if s.strip()]

    from orchestrator import run_discovery_orchestrator
    return run_discovery_orchestrator(student_id, ctx)


@router.post("/evidence")
def post_evidence(body: EvidenceIn) -> dict:
    world = _get_world()
    if body.node_id not in world.nodes:
        raise HTTPException(404, f"unknown graph node: {body.node_id}")
    try:
        ev = EvidenceObject(body.student_id, body.node_id,
                            edge_key=body.edge_key,
                            source=body.source,
                            outcome=body.outcome, confidence=body.confidence,
                            meta=body.meta)
    except ValueError as e:
        raise HTTPException(422, str(e))
    belief = twin.update(ev)
    resp = {"node_id": belief.node_id,
            "mastery": round(belief.mastery, 4),
            "exposures": belief.exposures,
            "mastered": belief.mastered}

    # Causal net: on a miss, diagnose WHY from the CASCADE edge beliefs and
    # name the edge to repair. On a correct edge, clear any open diagnosis.
    if body.outcome in ("incorrect", "partial"):
        try:
            import cause_net
            resp["diagnosis"] = cause_net.diagnose(
                body.student_id, body.node_id, world, outcome=body.outcome,
                l1=str(body.meta.get("l1", "en"))).to_dict()
        except Exception:
            pass
    elif body.outcome == "correct" and body.edge_key:
        try:
            import cause_net
            cause_net.resolve_edge(body.student_id, body.edge_key)
        except Exception:
            pass

    # Expedition arc: the PRIMARY evidence of a mission advances the journey.
    if body.source == "mission" and body.meta.get("primary"):
        import expedition as exp_mod
        exp = exp_mod.advance(body.student_id, body.node_id)
        if exp is not None:
            resp["expedition"] = {"display": exp.display, "step": exp.steps_done,
                                  "total": exp.steps_total, "status": exp.status}
            if exp.status == "complete":
                resp["world_unlocked"] = exp.display
    return resp


@router.get("/frontier/{student_id}")
def get_frontier(student_id: str, limit: int = 10) -> dict:
    world = _get_world()
    return {"frontier": [{
        "node_id": f.node_id, "display": f.display,
        "p_success": round(f.p_success, 3),
        "mastery": round(f.mastery, 3),
        "readiness": round(f.readiness, 3),
        "prerequisites": f.prerequisites,
    } for f in frontier(student_id, world, limit=limit)]}


# ── CASCADE Edge Endpoints ──────────────────────────────────────────

@router.get("/edge-frontier/{student_id}")
def get_edge_frontier(student_id: str, limit: int = 20) -> dict:
    world = _get_world()
    from development_engine import edge_frontier as ef
    return {"edge_frontier": [{
        "edge_key": f.edge_key, "source": f.source,
        "target": f.target, "etype": f.etype,
        "source_display": f.source_display, "target_display": f.target_display,
        "p_success": round(f.p_success, 3),
        "mastery": round(f.mastery, 3),
        "readiness": round(f.readiness, 3),
    } for f in ef(student_id, world, limit=limit)]}


@router.get("/diagnose/{student_id}")
def get_diagnose(student_id: str, node_id: str = Query(...),
                 l1: str = Query("en")) -> dict:
    """Causal net: the root-cause distribution for a miss on node_id, read off
    the learner's CASCADE edge beliefs. Names the CASCADE edge to repair."""
    import cause_net
    world = _get_world()
    if node_id not in world.nodes:
        raise HTTPException(404, f"unknown graph node: {node_id}")
    return cause_net.diagnose(student_id, node_id, world,
                              outcome="incorrect", l1=l1, persist=False).to_dict()


@router.get("/edge-twin/{student_id}")
def get_edge_twin(student_id: str) -> dict:
    world = _get_world()
    snap = twin.snapshot_edges(student_id)
    edges = sorted(snap.values(), key=lambda b: -b.mastery)
    return {"edges_tracked": len(edges),
            "mastered": sum(1 for b in edges if b.mastered),
            "top": [{"edge_key": b.edge_key,
                     "source_display": world.display(b.source),
                     "target_display": world.display(b.target),
                     "etype": b.etype,
                     "mastery": round(b.mastery, 3),
                     "exposures": b.exposures} for b in edges[:25]]}


@router.get("/twin/{student_id}")
def get_twin(student_id: str) -> dict:
    snap = twin.snapshot(student_id)
    world = _get_world()
    nodes = sorted(snap.values(), key=lambda b: -b.mastery)
    return {"nodes_tracked": len(nodes),
            "mastered": sum(1 for b in nodes if b.mastered),
            "top": [{"node_id": b.node_id,
                     "display": world.display(b.node_id),
                     "mastery": round(b.mastery, 3),
                     "exposures": b.exposures} for b in nodes[:25]]}


@router.get("/calibration/{student_id}")
def get_calibration(student_id: str) -> dict:
    return {"calibration": twin.calibration(student_id)}


@router.post("/calibration/tune")
def post_tune_calibration(student_id: str | None = Query(None)) -> dict:
    """Metacognitive Evaluation → parameter tuning.

    Reads the twin's calibration table across all students (or one)
    and adjusts the p_success blend weights so predicted probabilities
    stay honest. Called periodically or after evidence milestones."""
    from development_engine import tune_blend_from_calibration
    return tune_blend_from_calibration(student_id)


@router.get("/credal/{student_id}")
def get_credal(student_id: str) -> dict:
    """U_t — the credal (Beta) uncertainty view of the learner state.
    Per node: observable-accuracy mean, sd, effective evidence mass,
    expected information gain of one more observation (T5), and the
    BKT-vs-Beta misfit diagnostic."""
    import credal
    world = _get_world()
    snap = credal.snapshot(student_id)
    beliefs = twin.snapshot(student_id)
    nodes = sorted(snap.values(), key=lambda c: -c.eig)
    return {"nodes_tracked": len(nodes),
            "mean_sd": round(sum(c.sd for c in nodes) / len(nodes), 4) if nodes else None,
            "top_information_targets": [{
                "node_id": c.node_id,
                "display": world.display(c.node_id),
                "mean": round(c.mean, 3),
                "sd": round(c.sd, 3),
                "n_eff": round(c.n_eff, 2),
                "eig": round(c.eig, 5),
                "misfit": round(c.misfit(beliefs[c.node_id].mastery), 3)
                          if c.node_id in beliefs else None,
            } for c in nodes[:25]]}


@router.get("/pronounce/available")
def pronounce_available() -> dict:
    """SPA gate: show the mic only when the ear channel is configured."""
    import ear
    return {"available": bool(ear.EAR_URL)}


@router.get("/learner/{student_id}")
def get_learner_state(student_id: str) -> dict:
    """Unified Learner State — twin + credal + fingerprint + expedition
    in one structured response. No LLM, purely symbolic aggregation."""
    from learner_state import get_learner_state
    state = get_learner_state(student_id)
    return {"learner": state.to_dict()}


@router.post("/pronounce/{student_id}")
async def post_pronounce(
    student_id: str,
    audio: UploadFile = File(...),
    node_id: str = Form(...),
    target_text: str = Form(""),
    language: str = Form("en"),
    l1: str = Form("en"),
) -> dict:
    """Perception channel for V_Φ: prompted-speech clip → CAVP-lite engine →
    per-phone evidence → twin (BKT + credal). Honest abstention: a bad
    recording returns recorded=false and changes nothing."""
    import ear

    world = _get_world()
    if node_id not in world.nodes:
        raise HTTPException(404, f"unknown graph node: {node_id}")
    blob = await audio.read()
    if len(blob) > ear.MAX_AUDIO_BYTES:
        raise HTTPException(413, "audio clip too large")
    if not blob:
        raise HTTPException(422, "empty audio")

    target = target_text.strip() or world.display(node_id)
    try:
        result = ear.check_with_engine(blob, audio.filename or "clip.webm",
                                       target, language)
    except ear.EarUnavailable as e:
        raise HTTPException(503, str(e))
    return ear.ingest_pronunciation(student_id, node_id, result, world, l1=l1)


@router.get("/universe/{student_id}")
def get_universe(student_id: str) -> dict:
    """The child's personal constellation: every node they've met, its
    mastery, the edges among met nodes, unlocked worlds, active expedition."""
    import sqlite3
    import expedition as exp_mod
    from activity_generator import get_edges
    from evidence_graph import DB_PATH

    world = _get_world()
    with sqlite3.connect(DB_PATH) as c:
        met = [r[0] for r in c.execute(
            "SELECT DISTINCT node_id FROM evidence WHERE student_id=?",
            (student_id,)).fetchall()]
    met_set = set(met)
    snap = twin.snapshot(student_id)

    stars = [{"id": n, "display": world.display(n),
              "mastery": round(snap[n].mastery, 3) if n in snap else 0.1}
             for n in met if n in world.nodes]

    edges_out = []
    for etype, pairs in get_edges().by_type.items():
        for s, t in pairs:
            if s in met_set and t in met_set:
                edges_out.append({"a": s, "b": t, "type": etype})

    exp = exp_mod.get_active(student_id)
    return {"stars": stars, "links": edges_out,
            "worlds": exp_mod.unlocked_worlds(student_id),
            "active_expedition": ({"display": exp.display, "anchor": exp.anchor_id,
                                   "step": exp.steps_done + 1, "total": exp.steps_total}
                                  if exp else None)}
