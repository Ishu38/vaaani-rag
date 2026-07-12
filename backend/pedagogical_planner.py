"""Pedagogical Planning Engine — selects the next best learning activity.

Architecture position (Neil's diagram, 2026-07-12): reads the Development
Engine's frontier and the Cognitive Twin; emits ONE mission decision with an
explicit, human-readable reason. The Procedural Activity Generator then turns
the decision into learner-facing content (template + world-model facts; the
LLM only phrases, never chooses).

Formalism: decision theory (AIMA ch. 16) — myopic expected-utility
maximization over the belief state. No RL, no LLM in the decision path.

    score(node) = W_GAIN  * expected mastery gain      (BKT lookahead)
                + W_DECAY * review urgency              (forgetting curve)
                + W_NOVEL * novelty                     (exposure diversity)

Every decision is logged to the twin's prediction table (metacognition):
the planner says P(success) out loud before the learner attempts it, so the
Metacognitive Evaluation stage can later check whether the system's
confidence was honest.
"""

from __future__ import annotations

from dataclasses import dataclass

import cognitive_twin as twin
from cognitive_twin import TRANSIT, MASTERED_AT
from development_engine import WorldModel, frontier, p_success

W_GAIN, W_DECAY, W_NOVEL = 1.0, 0.6, 0.2
REVIEW_BELOW = 0.80          # mastered node decayed under this → review candidate

# v1 policy term (theory §6): expected information gain about the learner,
# from the credal channel (T5 closed form). DARK-SHIPPED: weight defaults to
# 0 so live pedagogy is unchanged until the term is calibrated — same
# discipline as the MLAF W1 rule (never flip an uncalibrated weight on).
import os
W_EIG = float(os.environ.get("VAAANI_POLICY_EIG", "0"))


@dataclass
class MissionDecision:
    kind: str                # "learn" | "review"
    node_id: str
    display: str
    p_success: float
    score: float
    reason: str              # explainable pedagogy, shown to teachers/parents


def _review_candidates(student_id: str, world: WorldModel) -> list[MissionDecision]:
    out = []
    for node_id, belief in twin.snapshot(student_id).items():
        if belief.exposures == 0 or node_id not in world.nodes:
            continue
        # node once mastered but decayed below review threshold
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


def select_activity(student_id: str, world: WorldModel | None = None) -> MissionDecision | None:
    """The planner's single job: argmax expected utility, with its reasoning.

    Expedition-aware: an active arc pins selection to its queue so the child
    experiences a journey; between arcs the tier-rotated scorer picks the
    next anchor and a new expedition begins.
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

    # Variety pressure: what the learner just did should not be what they do
    # next. Tier rotation (morph -> sound -> bridge -> ...) plus a hard-ish
    # penalty on repeating recent nodes — differentiated activities are an
    # architectural promise, not a nice-to-have.
    from activity_generator import node_tier
    from evidence_graph import recent_nodes
    recent = recent_nodes(student_id, source="mission", limit=6)
    recent_set = set(recent)
    recent_tiers = [node_tier(n) for n in recent[:3]]
    last_tier = recent_tiers[0] if recent_tiers else None

    # Tier-stratified candidate pool: the flat band-distance cut buries sound
    # and bridge nodes behind concept nodes (all fresh nodes tie on p_success).
    # Guarantee every linguistic tier a seat at the table, then let scoring
    # + rotation decide — differentiated assignments by construction.
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
            variety -= 0.08                      # just did this kind — rotate
        elif tier not in recent_tiers:
            variety += 0.05                      # a kind we haven't seen lately
        if f.node_id in recent_set:
            variety -= 0.12                      # same node again — strongly avoid
        score = W_GAIN * gain + W_NOVEL * novelty + variety
        if W_EIG:
            import credal
            # normalised: EIG max is 1/(N0+1)^2 at a fresh node with mu=0.5
            score += W_EIG * credal.get(student_id, f.node_id).eig * ((credal.N0 + 1) ** 2)
        candidates.append(MissionDecision(
            "learn", f.node_id, f.display, f.p_success, score,
            f"in ZPD (P(success)={f.p_success:.2f}), expected mastery gain "
            f"{gain:.2f}, readiness {f.readiness:.2f}; activity tier '{tier}'"
            + (" — rotating away from repeats" if variety > 0 else "")))

    candidates += _review_candidates(student_id, world)

    if not candidates:
        # Abstention / information gathering (metacognitive rule): the belief
        # state is too thin to place any node in the ZPD. Instead of guessing,
        # the agent PROBES — pick the least-evidenced structurally-connected
        # node and ask, explicitly to learn about the learner. This is the
        # agentic move: acting to reduce its own uncertainty.
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
    # A fresh "learn" pick becomes the anchor of a new expedition — the next
    # few missions stay inside its world.
    if best.kind == "learn":
        tier = node_tier(best.node_id)
        queue = _expedition_queue(best.node_id, tier, student_id, world)
        exp_mod.start(student_id, best.node_id, best.display, tier, queue)
    # metacognition: commit the prediction before the learner attempts it
    twin.log_prediction(student_id, best.node_id, best.p_success)
    return best
