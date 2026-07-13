"""Pedagogical Planning Engine — selects the next best learning activity.

Architecture position (Neil's diagram, 2026-07-12): reads the Development
Engine's frontier and the Cognitive Twin; emits ONE mission decision with an
explicit, human-readable reason.

CASCADE (2026-07-13): the planner now handles BOTH node-level and edge-level
candidates. Edge candidates are scored by structural importance × learnability,
preferring edges that connect the learner's known nodes into a larger
percolated component. Node candidates keep the existing BKT utility-max.
The argmax picks the single best across both pools.

Formalism: decision theory (AIMA ch. 16) — no RL, no LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

import cognitive_twin as twin
from cognitive_twin import TRANSIT, MASTERED_AT
from development_engine import (
    WorldModel, frontier, p_success,
    edge_frontier, p_success_edge, get_world_edges,
)

W_GAIN, W_DECAY, W_NOVEL = 1.0, 0.6, 0.2
REVIEW_BELOW = 0.80

# CASCADE edge-scoring weights.
# W_EDGE_STRUCTURE raised 0.4→0.7 (2026-07-13): the curvature-sequencing race
# (research/curvature_sequencing.py) showed easy-first is the WORST policy for
# reaching percolation; readiness-gated bridge-first wins (+0.07 AUPC). The
# readiness gate lives in edge_frontier_candidates, so weighting structure more
# steers toward learnable bridges without abandoning the ZPD.
W_EDGE_GAIN = 0.8
W_EDGE_STRUCTURE = 0.7
W_EDGE_NOVEL = 0.15
W_EDGE_DIAG = 0.6         # causal net implicated this edge as the repair target

import os
W_EIG = float(os.environ.get("VAAANI_POLICY_EIG", "0"))


@dataclass
class MissionDecision:
    kind: str                # "learn" | "review" | "edge_sequence" | "probe"
    node_id: str
    display: str
    p_success: float
    score: float
    reason: str
    edge_key: str = ""       # CASCADE: set when this is an edge mission


def _review_candidates(student_id: str, world: WorldModel) -> list[MissionDecision]:
    out = []
    for node_id, belief in twin.snapshot(student_id).items():
        if belief.exposures == 0 or node_id not in world.nodes:
            continue
        if belief.mastery < REVIEW_BELOW and belief.exposures >= 3:
            p, _, _ = p_success(student_id, node_id, world)
            urgency = REVIEW_BELOW - belief.mastery
            out.append(MissionDecision(
                "review", node_id, world.display(node_id), p,
                W_DECAY * urgency,
                f"mastery decayed to {belief.mastery:.2f} (below {REVIEW_BELOW}); "
                f"spaced review due"))
    return out


def _expedition_queue(anchor_id: str, tier: str, student_id: str,
                      world: WorldModel) -> list[str]:
    """Anchor + its unmastered same-tier neighbourhood — the arc's steps."""
    from activity_generator import get_edges
    edges = get_edges()
    tier_edges = {"morph": ("root_of", "word_family"),
                  "sound": ("sounds_like",),
                  "bridge": ("translates_to", "means"),
                  "other": ("used_in", "is_a")}[tier]
    related: list[str] = []
    for et in tier_edges:
        related += edges.neighbors(anchor_id, et)
    queue = [anchor_id]
    for n in related:
        if n in world.nodes and n not in queue and not twin.get(student_id, n).mastered:
            queue.append(n)
    return queue


def _edge_candidates(student_id: str, world: WorldModel,
                     recent: set[str]) -> list[MissionDecision]:
    """CASCADE: edge candidates with Ricci curvature + betweenness scoring.

    Closes the causal loop: if the cause-net has implicated a specific edge as
    the repair target for a recent miss, that edge is boosted (and injected if
    the frontier filter would otherwise hide it) so the plan acts on the
    diagnosis."""
    from edge_state import edge_frontier_candidates
    graph_edges = get_world_edges()
    pool = edge_frontier_candidates(
        student_id, graph_edges,
        node_display=world.display,
        limit=40,
    )
    # Causal-net repair target (a CASCADE edge), if any is open.
    diag_edge, diag_cause = "", ""
    try:
        import cause_net
        d = cause_net.latest_open(student_id)
        if d:
            diag_edge, diag_cause = d["edge_key"], d["top_cause"]
    except Exception:
        pass

    candidates: list[MissionDecision] = []
    seen_edges: set[str] = set()
    for f in pool:
        seen_edges.add(f.edge_key)
        gain = TRANSIT * (1.0 - f.mastery)
        novelty = 0.0 if twin.get_edge(student_id, f.edge_key).exposures else 1.0
        score = (W_EDGE_GAIN * gain +
                 W_EDGE_STRUCTURE * f.structural_importance +
                 W_EDGE_NOVEL * novelty)
        reason = (f"CASCADE: {f.source_display} —{f.etype}→ {f.target_display}; "
                  f"P(success)={f.p_success:.2f}, κ={f.curvature:.3f}, "
                  f"struct={f.structural_importance:.3f}")
        if f.edge_key == diag_edge:
            score += W_EDGE_DIAG
            reason += f"; repair target for diagnosed '{diag_cause}'"
        candidates.append(MissionDecision(
            "edge_sequence", f.source, f.source_display,
            f.p_success, score, reason, edge_key=f.edge_key))

    # Inject the diagnosed repair edge if the frontier filter hid it.
    if diag_edge and diag_edge not in seen_edges:
        parts = diag_edge.split("::")
        if len(parts) >= 3:
            s, t, et = parts[0], parts[1], parts[2]
            p, _, _, _, _ = p_success_edge(student_id, diag_edge, world)
            candidates.append(MissionDecision(
                "edge_sequence", s, world.display(s), p,
                W_EDGE_GAIN * TRANSIT + W_EDGE_DIAG,
                f"CASCADE repair: {world.display(s)} —{et}→ {world.display(t)}; "
                f"diagnosed '{diag_cause}' — mending the weak link directly",
                edge_key=diag_edge))
    return candidates


def select_activity(student_id: str, world: WorldModel | None = None) -> MissionDecision | None:
    """The planner's single job: argmax over node AND edge candidates.

    CASCADE: edge candidates compete with node candidates. An edge
    that connects the learner's known nodes into a larger percolated
    component can outscore a node candidate.
    """
    world = world or WorldModel()

    import expedition as exp_mod
    exp = exp_mod.get_active(student_id)
    if exp is not None and exp.queue:
        node_id = exp.queue[0]
        p, m, r = p_success(student_id, node_id, world)
        decision = MissionDecision(
            "learn", node_id, world.display(node_id), p, 1.0,
            f"expedition '{exp.display}' — step {exp.steps_done + 1} of "
            f"{exp.steps_total}; staying inside one world so discoveries "
            f"connect instead of scattering")
        twin.log_prediction(student_id, node_id, p)
        return decision

    from activity_generator import node_tier
    from evidence_graph import recent_nodes
    recent = recent_nodes(student_id, source="mission", limit=6)
    recent_set = set(recent)
    recent_tiers = [node_tier(n) for n in recent[:3]]
    last_tier = recent_tiers[0] if recent_tiers else None

    # CASCADE: nodes already percolated get deprioritized
    percolated: set[str] = set()
    try:
        percolated = twin.mastered_nodes_percolation(student_id)
    except Exception:
        pass

    # ── Node candidates ─────────────────────────────────────────
    pool = frontier(student_id, world, limit=600)
    by_tier: dict[str, list] = {}
    for f in pool:
        by_tier.setdefault(node_tier(f.node_id), []).append(f)
    stratified = []
    for tier_nodes in by_tier.values():
        stratified.extend(tier_nodes[:15])

    candidates: list[MissionDecision] = []
    for f in stratified:
        gain = TRANSIT * (1.0 - f.mastery)
        novelty = 0.0 if twin.get(student_id, f.node_id).exposures else 1.0
        tier = node_tier(f.node_id)
        variety = {"morph": 0.04, "sound": 0.06, "bridge": 0.06, "other": 0.0}[tier]
        if tier == last_tier:
            variety -= 0.08
        elif tier not in recent_tiers:
            variety += 0.05
        if f.node_id in recent_set:
            variety -= 0.12
        if f.node_id in percolated:
            variety -= 0.15  # CASCADE: already percolated — strongly deprioritize
        score = W_GAIN * gain + W_NOVEL * novelty + variety
        if W_EIG:
            import credal
            score += W_EIG * credal.get(student_id, f.node_id).eig * ((credal.N0 + 1) ** 2)
        candidates.append(MissionDecision(
            "learn", f.node_id, f.display, f.p_success, score,
            f"in ZPD (P(success)={f.p_success:.2f}), expected mastery gain "
            f"{gain:.2f}, readiness {f.readiness:.2f}; activity tier '{tier}'"
            + (" — rotating away from repeats" if variety > 0 else "")))

    candidates += _review_candidates(student_id, world)

    # ── CASCADE: Edge candidates ─────────────────────────────────
    candidates += _edge_candidates(student_id, world, recent_set)

    if not candidates:
        probe_pool = sorted(
            (n for n in world.nodes if world.prereqs.get(n)),
            key=lambda n: (twin.get(student_id, n).exposures, n)) or sorted(world.nodes)
        node_id = probe_pool[0]
        p, _, _ = p_success(student_id, node_id, world)
        decision = MissionDecision(
            "probe", node_id, world.display(node_id), p, 0.0,
            "belief state too thin to plan — probing to gather evidence "
            "instead of guessing")
        twin.log_prediction(student_id, node_id, p)
        return decision

    best = max(candidates, key=lambda c: c.score)
    if best.kind == "learn":
        tier = node_tier(best.node_id)
        queue = _expedition_queue(best.node_id, tier, student_id, world)
        exp_mod.start(student_id, best.node_id, best.display, tier, queue)
    elif best.kind == "edge_sequence":
        # Start a micro-expedition for the edge
        tier = node_tier(best.node_id)
        queue = [best.node_id]
        exp_mod.start(student_id, best.node_id,
                      f"{best.display}→{best.edge_key.split('::')[1]}",
                      tier, queue)
    twin.log_prediction(student_id, best.node_id, best.p_success)
    return best
