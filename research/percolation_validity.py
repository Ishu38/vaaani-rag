"""Predictive-validity test: does percolation (relational) mastery predict
TRANSFER better than node-level BKT?

The central CASCADE claim is that language mastery is a *topological* property
of the learned-edge subgraph, not a per-item scalar. If true, then for an item
a learner has NEVER directly practised, the state of the *edges around it*
should predict success better than the item's own BKT belief.

Design (built to be falsifiable, not rigged):

  1. Ground truth is an INDEPENDENT generative model with a tunable transfer
     strength λ. Each simulated learner practises a contiguous region of the
     REAL Vaaani graph; their latent per-edge competence diffuses to unpractised
     neighbours with strength λ (λ=0 ⇒ no transfer; λ→1 ⇒ competence is fully
     relational). Outcomes are sampled from that latent competence.

  2. Two families of *observable* estimators are fit from the SAME practice
     observations:
        • node_bkt  — standard BKT mastery blended with prerequisite readiness
                      (the field-standard baseline, in its strongest fair form).
        • cascade   — CASCADE edge-BKT: percolation (binary, the real
                      percolation.percolated_nodes) and the continuous
                      edge-neighbourhood belief.

  3. We score both on HELD-OUT items (never directly practised) and compute AUC
     / Brier against the sampled transfer outcomes, swept over λ.

  Honesty guards:
    • node_bkt gets prerequisite readiness, so it is not a strawman.
    • the λ=0 null: with no true transfer, cascade must NOT beat node_bkt on
      held-out items — otherwise the estimator is hallucinating structure.
    • in-sample (practised) AUC is reported too: node_bkt should win there,
      where direct evidence exists.

Reuses the real graph, the real percolation implementation, and the exact BKT
constants from cognitive_twin, so it tests the shipped logic. No LLM. Seeded.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from cognitive_twin import PRIOR, SLIP, GUESS, TRANSIT           # exact shipped constants
from percolation import percolated_nodes                          # the real implementation
from development_engine import get_world_edges, WorldModel

# planner's node predictor blend (development_engine.p_success defaults)
W_MASTERY, W_READINESS = 0.35, 0.65
THETA_EDGE = 0.90          # percolation edge-learned threshold (shipped default)
K_MIN = 3                  # percolation min component size (shipped default)


# ── observation model (shared link function) ────────────────────────
def obs_prob(competence: float) -> float:
    """Latent competence → P(correct). Same slip/guess link both sides use."""
    return GUESS + (1.0 - GUESS - SLIP) * max(0.0, min(1.0, competence))


def bkt_update(p: float, correct: bool, w: float = 1.0) -> float:
    """Identical to cognitive_twin.update / update_edge posterior + learn step."""
    if correct:
        post = p * (1 - SLIP) / (p * (1 - SLIP) + (1 - p) * GUESS + 1e-12)
    else:
        post = p * SLIP / (p * SLIP + (1 - p) * (1 - SLIP) + 1e-12)
    post = (1 - w) * p + w * post
    return post + (1 - post) * TRANSIT


class EB:
    """Minimal edge-belief object for percolation.percolated_nodes (needs .mastery)."""
    __slots__ = ("mastery",)
    def __init__(self, m: float): self.mastery = m


# ── graph loading ───────────────────────────────────────────────────
def load_graph():
    edges_by_type = get_world_edges()                    # {etype: [(s,t),...]}
    world = WorldModel()
    nodes = list(world.nodes)
    incident: dict[str, list[str]] = {}                  # node -> [edge_key]
    node_nbrs: dict[str, set[str]] = {}                  # node -> {node}
    edge_nbrs: dict[str, set[str]] = {}                  # edge_key -> {edge_key sharing a node}
    node_of_edge: dict[str, tuple[str, str]] = {}
    for et, pairs in edges_by_type.items():
        for s, t in pairs:
            ek = f"{s}::{t}::{et}"
            node_of_edge[ek] = (s, t)
            incident.setdefault(s, []).append(ek)
            incident.setdefault(t, []).append(ek)
            node_nbrs.setdefault(s, set()).add(t)
            node_nbrs.setdefault(t, set()).add(s)
    for n, eks in incident.items():
        for ek in eks:
            edge_nbrs.setdefault(ek, set()).update(e for e in eks if e != ek)
    prereqs = {n: world.prereqs.get(n, []) for n in nodes}
    return edges_by_type, nodes, incident, node_nbrs, edge_nbrs, node_of_edge, prereqs


# ── one simulated learner ───────────────────────────────────────────
def practised_region(nodes, node_nbrs, rng, density):
    """Contiguous practice: random-walk from a few seeds (curricula cluster)."""
    target = max(8, int(len(nodes) * density))
    practised: set[str] = set()
    seeds = rng.sample([n for n in nodes if node_nbrs.get(n)], k=min(5, len(nodes)))
    frontier = list(seeds)
    while frontier and len(practised) < target:
        cur = frontier.pop(rng.randrange(len(frontier)))
        if cur in practised:
            continue
        practised.add(cur)
        nbrs = list(node_nbrs.get(cur, ()))
        rng.shuffle(nbrs)
        frontier.extend(nbrs[:3])
    return practised


def simulate(G, rng, lam, density, attempts=3):
    edges_by_type, nodes, incident, node_nbrs, edge_nbrs, node_of_edge, prereqs = G
    alpha = rng.uniform(0.35, 0.9)                        # learner ability
    practised = practised_region(nodes, node_nbrs, rng, density)

    # Ground truth: NODE competence with λ-controlled transfer. A practised node
    # is acquired (α); an unpractised node's competence is ONLY what transfers
    # from its practised graph-neighbours, scaled by λ. λ=0 ⇒ unpractised nodes
    # are truly unknown (clean null — nothing is predictable). No edge-sharing
    # leakage: λ is the sole transfer knob.
    def true_comp(v):
        if v in practised:
            return alpha
        nb = node_nbrs.get(v, ())
        if not nb:
            return 0.0
        frac = sum(1 for u in nb if u in practised) / len(nb)
        return lam * alpha * frac

    # ── fit estimators from practice observations only ──
    node_bel = {}
    edge_bel = {}
    for v in practised:
        pv = obs_prob(true_comp(v))
        nb = PRIOR
        for _ in range(attempts):
            correct = rng.random() < pv
            nb = bkt_update(nb, correct)
            for ek in incident.get(v, []):               # item attempt updates incident edges (ear×CASCADE)
                edge_bel[ek] = bkt_update(edge_bel.get(ek, PRIOR), correct)
        node_bel[v] = nb

    # percolation set from the fitted edge beliefs (REAL implementation)
    eb_objs = {ek: EB(m) for ek, m in edge_bel.items()}
    perc = percolated_nodes(eb_objs, edges_by_type, theta=THETA_EDGE, k_min=K_MIN)

    # ── score held-out items (never practised) ──
    rows = []
    held = [v for v in nodes if v not in practised and incident.get(v)]
    rng.shuffle(held)
    for v in held[:400]:                                 # cap per learner for speed
        outcome = 1 if rng.random() < obs_prob(true_comp(v)) else 0
        # (1) node_bkt: prior mastery + PREREQUISITE readiness (field-standard baseline)
        pr = prereqs.get(v, [])
        readiness = statistics.fmean([node_bel.get(x, PRIOR) for x in pr]) if pr else PRIOR
        s_node = GUESS + (1 - GUESS - SLIP) * (W_MASTERY * PRIOR + W_READINESS * readiness)
        # (2) node_spread: prior blended with mean node-belief over ALL neighbours
        #     (the tough baseline — "why not just average the neighbours' node masteries?")
        nb = node_nbrs.get(v, ())
        spread = statistics.fmean([node_bel.get(u, PRIOR) for u in nb]) if nb else PRIOR
        s_spread = GUESS + (1 - GUESS - SLIP) * (W_MASTERY * PRIOR + W_READINESS * spread)
        # (3) cascade edge-neighbourhood belief
        eks = incident.get(v, [])
        nbr_edge = statistics.fmean([edge_bel.get(e, PRIOR) for e in eks]) if eks else PRIOR
        s_cascade = obs_prob(nbr_edge)
        # (4) percolation (binary)
        s_perc = 1.0 if v in perc else 0.0
        rows.append((outcome, s_node, s_spread, s_cascade, s_perc))
    return rows


# ── metrics ─────────────────────────────────────────────────────────
def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    sum_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def brier(scores, labels):
    return statistics.fmean([(s - y) ** 2 for s, y in zip(scores, labels)])


# ── experiment ──────────────────────────────────────────────────────
def run(n_students=200, density=0.12, lambdas=(0.0, 0.2, 0.4, 0.6, 0.8), seed=7):
    G = load_graph()
    results = {}
    for lam in lambdas:
        rng = random.Random(seed * 1000 + int(lam * 100))
        y, sn, ss, sc, sp = [], [], [], [], []
        for _ in range(n_students):
            for outcome, s_node, s_spread, s_casc, s_perc in simulate(G, rng, lam, density):
                y.append(outcome); sn.append(s_node); ss.append(s_spread)
                sc.append(s_casc); sp.append(s_perc)
        base_rate = statistics.fmean(y) if y else float("nan")
        results[lam] = {
            "n_heldout": len(y),
            "held_out_base_rate": round(base_rate, 4),
            "auc_node_bkt": round(auc(sn, y), 4),
            "auc_node_spread": round(auc(ss, y), 4),
            "auc_cascade_edge": round(auc(sc, y), 4),
            "auc_percolation": round(auc(sp, y), 4),
            "brier_node_bkt": round(brier(sn, y), 4),
            "brier_cascade_edge": round(brier(sc, y), 4),
            "auc_lift_cascade_over_node": round(auc(sc, y) - auc(sn, y), 4),
            "auc_lift_cascade_over_spread": round(auc(sc, y) - auc(ss, y), 4),
        }
    return results


def simulate_edge_specific(G, rng, lam, density, attempts=3):
    """Second regime: competence is EDGE-specific (a learner can know the
    phone→/f/ relation without fully knowing 'phone' as an item). Practice is at
    the edge level; the held-out target is an unpractised EDGE. This is where
    edge-level tracing should beat node-level summaries, because a node's mastery
    aggregates ALL its edges and blurs any single relation."""
    edges_by_type, nodes, incident, node_nbrs, edge_nbrs, node_of_edge, prereqs = G
    alpha = rng.uniform(0.35, 0.9)
    all_edges = list(node_of_edge)
    # contiguous edge practice: random walk over edge-adjacency
    target = max(8, int(len(all_edges) * density))
    practised: set[str] = set()
    frontier = rng.sample(all_edges, k=min(5, len(all_edges)))
    while frontier and len(practised) < target:
        cur = frontier.pop(rng.randrange(len(frontier)))
        if cur in practised:
            continue
        practised.add(cur)
        nb = list(edge_nbrs.get(cur, ()))
        rng.shuffle(nb)
        frontier.extend(nb[:3])

    def true_edge_comp(e):
        if e in practised:
            return alpha
        nb = edge_nbrs.get(e, ())
        if not nb:
            return 0.0
        frac = sum(1 for x in nb if x in practised) / len(nb)
        return lam * alpha * frac

    node_bel, edge_bel = {}, {}
    for e in practised:
        pe = obs_prob(true_edge_comp(e))
        eb = PRIOR
        s, t = node_of_edge[e]
        nb_s, nb_t = node_bel.get(s, PRIOR), node_bel.get(t, PRIOR)
        for _ in range(attempts):
            correct = rng.random() < pe
            eb = bkt_update(eb, correct)
            nb_s = bkt_update(nb_s, correct)     # edge practice trains both endpoint nodes
            nb_t = bkt_update(nb_t, correct)
        edge_bel[e] = eb
        node_bel[s], node_bel[t] = nb_s, nb_t

    rows = []
    held = [e for e in all_edges if e not in practised]
    rng.shuffle(held)
    for e in held[:400]:
        outcome = 1 if rng.random() < obs_prob(true_edge_comp(e)) else 0
        s, t = node_of_edge[e]
        # node baseline for an edge = mean of endpoint node masteries (p_success_edge readiness)
        s_node = obs_prob((node_bel.get(s, PRIOR) + node_bel.get(t, PRIOR)) / 2)
        # node-smoothing: endpoints' neighbourhood node masteries
        ns = [node_bel.get(u, PRIOR) for u in (node_nbrs.get(s, ()) | node_nbrs.get(t, ()))]
        s_spread = obs_prob(statistics.fmean(ns) if ns else PRIOR)
        # cascade: mean belief over ADJACENT edges (edge-level neighbourhood)
        enb = edge_nbrs.get(e, ())
        s_cascade = obs_prob(statistics.fmean([edge_bel.get(x, PRIOR) for x in enb]) if enb else PRIOR)
        rows.append((outcome, s_node, s_spread, s_cascade))
    return rows


def run_edge_specific(n_students=200, density=0.12, lambdas=(0.0, 0.2, 0.4, 0.6, 0.8), seed=11):
    G = load_graph()
    results = {}
    for lam in lambdas:
        rng = random.Random(seed * 1000 + int(lam * 100))
        y, sn, ss, sc = [], [], [], []
        for _ in range(n_students):
            for outcome, s_node, s_spread, s_casc in simulate_edge_specific(G, rng, lam, density):
                y.append(outcome); sn.append(s_node); ss.append(s_spread); sc.append(s_casc)
        results[lam] = {
            "n_heldout": len(y),
            "auc_node_bkt": round(auc(sn, y), 4),
            "auc_node_spread": round(auc(ss, y), 4),
            "auc_cascade_edge": round(auc(sc, y), 4),
            "auc_lift_cascade_over_spread": round(auc(sc, y) - auc(ss, y), 4),
        }
    return results


def main():
    res = run()
    print(f"\n{'λ':>4}  {'n':>7}  {'base':>6}  | {'node-BKT':>9} {'node-smth':>9} "
          f"{'cascade':>9} {'percol':>9} | {'lift/node':>9} {'lift/smth':>9}")
    print("-" * 96)
    for lam, r in res.items():
        print(f"{lam:>4.1f}  {r['n_heldout']:>7}  {r['held_out_base_rate']:>6.3f}  | "
              f"{r['auc_node_bkt']:>9.3f} {r['auc_node_spread']:>9.3f} "
              f"{r['auc_cascade_edge']:>9.3f} {r['auc_percolation']:>9.3f} | "
              f"{r['auc_lift_cascade_over_node']:>+9.3f} {r['auc_lift_cascade_over_spread']:>+9.3f}")
    # ── Regime 2: edge-specific competence (the distinctive CASCADE claim) ──
    res2 = run_edge_specific()
    print(f"\nEDGE-SPECIFIC regime (held-out target = an unpractised relation):")
    print(f"{'λ':>4}  {'n':>7}  | {'node-BKT':>9} {'node-smth':>9} {'cascade':>9} | {'lift/smth':>9}")
    print("-" * 62)
    for lam, r in res2.items():
        print(f"{lam:>4.1f}  {r['n_heldout']:>7}  | {r['auc_node_bkt']:>9.3f} "
              f"{r['auc_node_spread']:>9.3f} {r['auc_cascade_edge']:>9.3f} | "
              f"{r['auc_lift_cascade_over_spread']:>+9.3f}")

    out = Path(__file__).resolve().parent / "percolation_validity_results.json"
    out.write_text(json.dumps({"node_level_regime": res, "edge_specific_regime": res2}, indent=2))
    print(f"\nwrote {out}")

    # optional plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        L = list(res.keys())
        plt.figure(figsize=(7, 4.3))
        plt.plot(L, [res[l]["auc_node_bkt"] for l in L], "o-", label="node-BKT (+prereq readiness)", color="#8a8a97")
        plt.plot(L, [res[l]["auc_node_spread"] for l in L], "^:", label="node-neighbour smoothing", color="#c98a12")
        plt.plot(L, [res[l]["auc_cascade_edge"] for l in L], "o-", label="CASCADE edge-neighbourhood", color="#4f46e5")
        plt.plot(L, [res[l]["auc_percolation"] for l in L], "s--", label="CASCADE percolation (binary)", color="#0f9d6b")
        plt.axhline(0.5, color="#ccc", lw=1, ls=":")
        plt.xlabel("transfer strength λ (ground truth)")
        plt.ylabel("AUC on held-out (unpractised) items")
        plt.title("Predicting transfer: relational vs node mastery")
        plt.legend(fontsize=9); plt.grid(alpha=.25); plt.tight_layout()
        p = Path(__file__).resolve().parent / "percolation_validity.png"
        plt.savefig(p, dpi=140)
        print(f"wrote {p}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
