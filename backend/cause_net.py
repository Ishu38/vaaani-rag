"""Causal Diagnostic Net — the "why" behind a miss, reading CASCADE.

Architecture position (Neil's diagram): the Causal Reasoner box. When the
Cognitive Twin records a wrong/partial outcome, *something* caused it. This
module infers a probability distribution over ROOT CAUSES — not a single
label — from evidence the rest of the system already holds. Zero LLM: a small
naive-Bayes network (AIMA ch. 13) with a documented conditional-probability
table and L1-tilted priors.

Relation to CASCADE (this is the point): the observations are read straight
off the CASCADE edge beliefs. A miss on the node "phone" is diagnosed by
looking at how solid the child's *edges* are — phone—sounds_like→f,
f—written_as→ph. If that phoneme↔grapheme edge is weak, the net raises
P(phoneme_grapheme_gap) and names THAT edge as the one to repair. The named
edge is a CASCADE edge, so the Pedagogical Planner can target it directly and
CASCADE propagation carries the repair to its neighbours.

Causes (hidden):
    l1_interference        home-language sound/structure bleeding through
    phoneme_grapheme_gap   the sound↔spelling link is not solid
    insufficient_exposure  simply not met enough times yet
    prerequisite_gap       something this builds on is shaky
    careless_slip          knows it — a lapse, not a gap

Observations (binary, derived from twin + CASCADE):
    E_low   the implicated CASCADE edge belief < EDGE_WEAK
    P_low   mean prerequisite node mastery < PREREQ_WEAK
    L1      an L1 script / cognate is involved AND l1 != 'en'
    FEW     the node has been met < FEW_EXPOSURES times
    HIGH    node mastery >= SLIP_MASTERY (so a miss reads as a slip)

Everything is deterministic, CPU-only, and reproducible. The CPT below is a
v0 literature-informed prior, not fitted — flagged as such in the report.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import cognitive_twin as twin
from cognitive_twin import DB_PATH
from l1_graft import L1_NAMES, l1_boost_factor, _is_l1_script

# ── Observation thresholds ──────────────────────────────────────────
EDGE_WEAK = 0.40
PREREQ_WEAK = 0.40
FEW_EXPOSURES = 3
SLIP_MASTERY = 0.70

CAUSES = [
    "l1_interference",
    "phoneme_grapheme_gap",
    "insufficient_exposure",
    "prerequisite_gap",
    "careless_slip",
]

# Base priors P(cause) before the L1 tilt (sum to 1).
BASE_PRIOR = {
    "l1_interference": 0.20,
    "phoneme_grapheme_gap": 0.20,
    "insufficient_exposure": 0.25,
    "prerequisite_gap": 0.20,
    "careless_slip": 0.15,
}

# Conditional probabilities P(observation = True | cause). v0, documented.
# Rows = cause, columns = observation.  (AIMA naive-Bayes likelihoods.)
CPT = {
    #                    E_low  P_low  L1    FEW   HIGH
    "l1_interference":     {"E_low": 0.60, "P_low": 0.30, "L1": 0.90, "FEW": 0.40, "HIGH": 0.25},
    "phoneme_grapheme_gap":{"E_low": 0.90, "P_low": 0.30, "L1": 0.40, "FEW": 0.50, "HIGH": 0.20},
    "insufficient_exposure":{"E_low": 0.55, "P_low": 0.40, "L1": 0.30, "FEW": 0.90, "HIGH": 0.10},
    "prerequisite_gap":    {"E_low": 0.50, "P_low": 0.90, "L1": 0.30, "FEW": 0.50, "HIGH": 0.10},
    "careless_slip":       {"E_low": 0.20, "P_low": 0.20, "L1": 0.20, "FEW": 0.20, "HIGH": 0.90},
}

OBS_KEYS = ["E_low", "P_low", "L1", "FEW", "HIGH"]

# Plain-language, school-facing (any speaker) — no jargon as primary.
CAUSE_COPY = {
    "l1_interference": (
        "This looks like your home language nudging the sound — very common "
        "and completely fixable. We'll put the two side by side so the "
        "difference is easy to hear."),
    "phoneme_grapheme_gap": (
        "The link between the sound and how it's written isn't solid yet. "
        "We'll practise that exact sound–spelling pair."),
    "insufficient_exposure": (
        "You simply haven't met this enough times yet. A little more practice "
        "will lock it in."),
    "prerequisite_gap": (
        "Something earlier that this builds on is still a bit shaky. We'll "
        "shore that up first so this comes easily."),
    "careless_slip": (
        "You actually know this — it looks like a slip, not a gap. We'll keep "
        "moving and let it come back at the right time."),
}


@dataclass
class CauseDiagnosis:
    node_id: str
    outcome: str
    top_cause: str
    top_prob: float
    distribution: dict[str, float]          # cause -> posterior
    implicated_edge: str                    # CASCADE edge to repair ("" if none)
    implicated_edge_display: str
    observations: dict[str, bool]
    explanation: str                        # plain language
    l1: str = "en"
    substitution: dict | None = None        # contrastive L1 confusion, if any
    is_valid: bool = True
    note: str = field(default=(
        "v0 causal net: priors literature-informed, not cohort-fitted; "
        "observations read from live CASCADE edge beliefs."))

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "outcome": self.outcome,
            "top_cause": self.top_cause,
            "top_prob": round(self.top_prob, 3),
            "distribution": {c: round(p, 3) for c, p in self.distribution.items()},
            "implicated_edge": self.implicated_edge,
            "implicated_edge_display": self.implicated_edge_display,
            "observations": self.observations,
            "explanation": self.explanation,
            "l1": self.l1,
            "l1_name": L1_NAMES.get(self.l1, self.l1),
            "substitution": self.substitution,
            "note": self.note,
        }


def _conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS diagnoses (
        student_id TEXT NOT NULL,
        node_id    TEXT NOT NULL,
        edge_key   TEXT NOT NULL DEFAULT '',
        top_cause  TEXT NOT NULL,
        top_prob   REAL NOT NULL,
        ts         REAL NOT NULL,
        resolved   INTEGER NOT NULL DEFAULT 0
    )""")
    c.execute("CREATE INDEX IF NOT EXISTS ix_diag_student "
              "ON diagnoses(student_id, resolved, ts)")
    return c


def _incident_edges(node_id: str, graph_edges: dict) -> list[str]:
    """All CASCADE edge_keys touching this node, both directions."""
    out: list[str] = []
    for etype, pairs in graph_edges.items():
        for s, t in pairs:
            if s == node_id or t == node_id:
                out.append(f"{s}::{t}::{etype}")
    return out


def _weakest_incident_edge(student_id: str, node_id: str, graph_edges: dict
                           ) -> tuple[str, float]:
    """The CASCADE edge to repair: lowest-belief edge incident to the node.
    Returns (edge_key, mastery); ("", 1.0) when the node has no edges."""
    weakest, weakest_m = "", 1.0
    for ek in _incident_edges(node_id, graph_edges):
        m = twin.get_edge(student_id, ek).mastery
        if m < weakest_m:
            weakest, weakest_m = ek, m
    return weakest, weakest_m


def _edge_display(edge_key: str, world) -> str:
    if not edge_key:
        return ""
    parts = edge_key.split("::")
    if len(parts) < 3:
        return edge_key
    s, t, et = parts[0], parts[1], parts[2]
    return f"{world.display(s)} —{et}→ {world.display(t)}"


def _observe(student_id: str, node_id: str, world, l1: str,
             implicated_edge: str, edge_mastery: float) -> dict[str, bool]:
    node_b = twin.get(student_id, node_id)
    prereqs = world.prereqs.get(node_id, []) if hasattr(world, "prereqs") else []
    if prereqs:
        prereq_m = sum(twin.get(student_id, p).mastery for p in prereqs) / len(prereqs)
    else:
        prereq_m = 1.0

    # L1 involvement: the node OR either endpoint of the implicated edge is in
    # the L1 script (a cognate/translation node), and the learner has an L1.
    l1_involved = False
    if l1 and l1 != "en":
        disp = [world.display(node_id)]
        if implicated_edge:
            for nid in implicated_edge.split("::")[:2]:
                disp.append(world.display(nid))
        l1_involved = any(_is_l1_script(d, l1) for d in disp) or \
            ("::translates_to" in implicated_edge or "::cognate_with" in implicated_edge)

    return {
        "E_low": bool(implicated_edge) and edge_mastery < EDGE_WEAK,
        "P_low": prereq_m < PREREQ_WEAK,
        "L1": l1_involved,
        "FEW": node_b.exposures < FEW_EXPOSURES,
        "HIGH": node_b.mastery >= SLIP_MASTERY,
    }


def _priors(l1: str) -> dict[str, float]:
    """L1 tilts the prior toward interference for non-English home languages."""
    p = dict(BASE_PRIOR)
    if l1 and l1 != "en":
        # l1_boost_factor is 1.0 (en) .. 1.5 (bn/hi); reuse it as the tilt.
        p["l1_interference"] *= l1_boost_factor(l1)
    z = sum(p.values())
    return {c: v / z for c, v in p.items()}


def _posterior(obs: dict[str, bool], l1: str) -> dict[str, float]:
    """Naive-Bayes: P(C|O) ∝ P(C) ∏_o P(o|C)^[o] (1-P(o|C))^[¬o]."""
    prior = _priors(l1)
    scores: dict[str, float] = {}
    for c in CAUSES:
        s = prior[c]
        for k in OBS_KEYS:
            p = CPT[c][k]
            s *= p if obs[k] else (1.0 - p)
        scores[c] = s
    z = sum(scores.values()) or 1.0
    return {c: scores[c] / z for c in CAUSES}


def diagnose(student_id: str, node_id: str, world, outcome: str = "incorrect",
             l1: str = "en", persist: bool = True) -> CauseDiagnosis:
    """Infer the root-cause distribution for a miss on `node_id`.

    Reads CASCADE edge beliefs to build observations, runs the Bayes net,
    names the weakest incident CASCADE edge as the repair target, and (by
    default) persists the top diagnosis so the planner can act on it.
    """
    from development_engine import get_world_edges
    graph_edges = get_world_edges()

    implicated_edge, edge_m = _weakest_incident_edge(student_id, node_id, graph_edges)
    obs = _observe(student_id, node_id, world, l1, implicated_edge, edge_m)

    # Contrastive confusion: a live, quantitative L1 signal. If the node (or the
    # implicated edge) touches a phoneme this L1 substitutes, read the learner's
    # substitution belief and let it drive the l1_interference observation.
    sub = _find_confusion(student_id, l1, node_id, implicated_edge, world)
    if sub and sub["belief"] >= 0.50:
        obs["L1"] = True

    dist = _posterior(obs, l1)
    top_cause = max(dist, key=dist.get)

    # A slip diagnosis shouldn't send them to repair an edge.
    edge_out = "" if top_cause == "careless_slip" else implicated_edge

    explanation = CAUSE_COPY[top_cause]
    if top_cause == "l1_interference" and sub:
        explanation = (
            f"This looks like the {sub['target_ipa']} → {sub['attractor_ipa']} swap "
            f"that {L1_NAMES.get(l1, l1)} speakers make when English uses a sound your "
            f"home language doesn't — very common, very fixable. We'll put the two "
            f"sounds side by side so the difference is easy to hear.")

    diag = CauseDiagnosis(
        node_id=node_id,
        outcome=outcome,
        top_cause=top_cause,
        top_prob=dist[top_cause],
        distribution=dist,
        implicated_edge=edge_out,
        implicated_edge_display=_edge_display(edge_out, world),
        observations=obs,
        explanation=explanation,
        l1=l1,
        substitution=sub,
    )
    if persist:
        _persist(student_id, diag)
    return diag


def _find_confusion(student_id: str, l1: str, node_id: str,
                    implicated_edge: str, world) -> dict | None:
    """The learner's live substitution belief for any confused phoneme touched by
    this miss (the node itself or an endpoint of the implicated edge)."""
    if not l1 or l1 == "en":
        return None
    import l1_confusion as lc
    cands = [node_id]
    if implicated_edge:
        cands += implicated_edge.split("::")[:2]
    for n in cands:
        cb = lc.get(student_id, l1, n)
        if cb:
            return {"target": cb.target, "attractor": cb.attractor,
                    "target_ipa": world.display(cb.target),
                    "attractor_ipa": world.display(cb.attractor),
                    "belief": round(cb.belief, 3)}
    return None


def _persist(student_id: str, diag: CauseDiagnosis) -> None:
    now = time.time()
    with _conn() as c:
        # supersede prior open diagnoses for the same node
        c.execute("UPDATE diagnoses SET resolved=1 WHERE student_id=? AND node_id=?",
                  (student_id, diag.node_id))
        c.execute("INSERT INTO diagnoses VALUES (?,?,?,?,?,?,0)",
                  (student_id, diag.node_id, diag.implicated_edge,
                   diag.top_cause, diag.top_prob, now))


def latest_open(student_id: str) -> dict | None:
    """The most recent unresolved diagnosis with an implicated CASCADE edge.
    The Pedagogical Planner reads this to prioritise the repair edge."""
    with _conn() as c:
        row = c.execute(
            "SELECT node_id, edge_key, top_cause, top_prob, ts FROM diagnoses "
            "WHERE student_id=? AND resolved=0 AND edge_key!='' "
            "ORDER BY ts DESC LIMIT 1", (student_id,)).fetchone()
    if not row:
        return None
    return {"node_id": row[0], "edge_key": row[1], "top_cause": row[2],
            "top_prob": row[3], "ts": row[4]}


def resolve_edge(student_id: str, edge_key: str) -> None:
    """Mark diagnoses targeting this edge as resolved (planner acted / mastered)."""
    with _conn() as c:
        c.execute("UPDATE diagnoses SET resolved=1 "
                  "WHERE student_id=? AND edge_key=?", (student_id, edge_key))
