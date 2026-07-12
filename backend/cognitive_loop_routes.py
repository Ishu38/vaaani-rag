"""FastAPI router for the cognitive loop — /loop/* endpoints.

The whole architecture over HTTP, no LLM/SLM anywhere in the path:

  GET  /loop/mission/{student_id}      planner decision + generated activity
  POST /loop/evidence                  learner outcome -> twin update
  GET  /loop/frontier/{student_id}     Development Engine's ZPD frontier
  GET  /loop/twin/{student_id}         belief-state summary
  GET  /loop/calibration/{student_id}  Metacognitive Evaluation table

v0 note: student_id is taken from the path (same trust level as existing
/learning endpoints' session flow should be added when merging with auth).
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
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
    source: str = Field(description=f"one of {SOURCES}")
    outcome: str = Field(description=f"one of {OUTCOMES}")
    confidence: float = 1.0
    meta: dict = Field(default_factory=dict)


@router.get("/mission/{student_id}")
def get_mission(student_id: str) -> dict:
    import expedition as exp_mod
    from companion import companion_block

    world = _get_world()
    decision = select_activity(student_id, world)
    activity = generate(decision, world)
    exp = exp_mod.get_active(student_id)
    resp = {"decision": {
                "kind": decision.kind, "node_id": decision.node_id,
                "display": decision.display,
                "p_success": round(decision.p_success, 3),
                "reason": decision.reason},
            "activity": activity.to_dict(),
            "companion": companion_block(student_id, exp)}
    if exp is not None:
        resp["expedition"] = {"anchor": exp.anchor_id, "display": exp.display,
                              "tier": exp.tier, "step": exp.steps_done + 1,
                              "total": exp.steps_total, "status": exp.status}
    return resp


@router.post("/evidence")
def post_evidence(body: EvidenceIn) -> dict:
    world = _get_world()
    if body.node_id not in world.nodes:
        raise HTTPException(404, f"unknown graph node: {body.node_id}")
    try:
        ev = EvidenceObject(body.student_id, body.node_id, body.source,
                            body.outcome, body.confidence, meta=body.meta)
    except ValueError as e:
        raise HTTPException(422, str(e))
    belief = twin.update(ev)
    resp = {"node_id": belief.node_id,
            "mastery": round(belief.mastery, 4),
            "exposures": belief.exposures,
            "mastered": belief.mastered}

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


@router.post("/pronounce/{student_id}")
async def post_pronounce(
    student_id: str,
    audio: UploadFile = File(...),
    node_id: str = Form(...),
    target_text: str = Form(""),
    language: str = Form("en"),
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
    return ear.ingest_pronunciation(student_id, node_id, result, world)


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
