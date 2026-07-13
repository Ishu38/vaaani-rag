"""Curvature-guided sequencing: does teaching bridge edges first reach
whole-graph percolation faster than teaching the easy ones?

CASCADE defines mastery topologically: a node is mastered when its learned-edge
subgraph percolates (component >= k_min). So the *fastest* route to a mastered
learner is the fewest taught edges that connect the knowledge graph. Percolation
theory says the edges that merge components are the **bridges** — negatively
curved / high-betweenness edges. Hypothesis: sequence bridges first.

The honest tension this must respect: bridges span *unknown* communities, so
they are HARDER to learn (low readiness). A fair test therefore teaches under a
realistic learnability model — an edge only becomes "learned" when repeated
practice pushes its BKT belief past theta, and practice succeeds in proportion
to endpoint mastery. If bridge-first still wins despite lower learnability, the
policy is real; if it loses, we learn that bridges must be readiness-gated.

Policies raced (each picks the next edge to teach):
  random      — unlearned edge at random
  easy        — highest endpoint readiness (naive utility-max / lowest-hanging)
  betweenness — highest edge-betweenness (gold-standard bridge signal)
  curvature   — most negative shipped Forman-Ricci (the product's local signal)
  hybrid      — highest betweenness AMONG readiness>=gate edges (teach learnable bridges)

Metric: fraction of nodes percolated vs. teaching steps → AUPC (area under the
percolation curve; higher = faster) and steps-to-50%/80% coverage. Paired across
policies (same learner, same seeds) for variance control. Real graph, shipped
percolation + curvature + BKT. Seeded.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from cognitive_twin import PRIOR, SLIP, GUESS, TRANSIT
from percolation import percolated_nodes
from development_engine import get_world_edges
from ricci_curvature import get_curvature

THETA, K_MIN = 0.90, 3
ACCESS_FLOOR = 0.12        # an edge is teachable if >=1 endpoint is at least this known
READINESS_GATE = 0.30      # hybrid: only bridges with readiness >= this
ATTEMPTS = 3               # practice attempts per teaching step
WORLD_SIZE = 140           # tractable connected sub-world (BFS ball) per the experiment
BUDGET = 260               # teaching steps per learner
SEED_NODES = 3


class EB:
    __slots__ = ("mastery",)
    def __init__(self, m): self.mastery = m


def obs_prob(c): return GUESS + (1 - GUESS - SLIP) * max(0.0, min(1.0, c))

def bkt(p, correct):
    if correct:
        post = p * (1 - SLIP) / (p * (1 - SLIP) + (1 - p) * GUESS + 1e-12)
    else:
        post = p * SLIP / (p * SLIP + (1 - p) * (1 - SLIP) + 1e-12)
    return post + (1 - post) * TRANSIT


def _bfs_ball(full_nbrs, start, size):
    """A connected sub-world of `size` nodes (BFS from a well-connected hub)."""
    seen = {start}; frontier = [start]
    while frontier and len(seen) < size:
        nxt = []
        for n in frontier:
            for m in sorted(full_nbrs.get(n, ()), key=lambda x: -len(full_nbrs.get(x, ()))):
                if m not in seen:
                    seen.add(m); nxt.append(m)
                    if len(seen) >= size:
                        break
            if len(seen) >= size:
                break
        frontier = nxt
    return seen


def load():
    edges_by_type_full = get_world_edges()
    full_nbrs = {}
    for et, pairs in edges_by_type_full.items():
        for u, v in pairs:
            full_nbrs.setdefault(u, set()).add(v)
            full_nbrs.setdefault(v, set()).add(u)
    # pick the most-connected node, grow a tractable connected sub-world around it
    hub = max(full_nbrs, key=lambda n: len(full_nbrs[n]))
    world = _bfs_ball(full_nbrs, hub, WORLD_SIZE)

    edges_by_type = {}
    edges = []
    node_nbrs = {}
    for et, pairs in edges_by_type_full.items():
        for u, v in pairs:
            if u in world and v in world:
                ek = f"{u}::{v}::{et}"
                edges_by_type.setdefault(et, []).append((u, v))
                edges.append((ek, u, v))
                node_nbrs.setdefault(u, set()).add(v)
                node_nbrs.setdefault(v, set()).add(u)
    nodes = list(node_nbrs)
    # edge betweenness on the sub-world (gold-standard bridge signal, computed in-world)
    import networkx as nx
    g = nx.Graph()
    for _, u, v in edges:
        g.add_edge(u, v)
    ebc = nx.edge_betweenness_centrality(g)
    betw = {ek: ebc.get((u, v), ebc.get((v, u), 0.0)) for ek, u, v in edges}
    curv = {ek: get_curvature(ek) for ek, _, _ in edges}   # SHIPPED full-graph Forman [0,1]
    return edges_by_type, edges, nodes, node_nbrs, betw, curv


class Learner:
    """Shared simulation state; a policy is a scoring function over teachable edges."""
    def __init__(self, G, rng, alpha):
        self.G = G; self.rng = rng; self.alpha = alpha
        self.node_m = {}
        self.edge_b = {}
        _, edges, nodes, node_nbrs, _, _ = G
        for s in rng.sample([n for n in nodes if node_nbrs.get(n)], k=SEED_NODES):
            self.node_m[s] = 0.5                 # a few anchor nodes to start from

    def readiness(self, u, v):
        return (self.node_m.get(u, PRIOR) + self.node_m.get(v, PRIOR)) / 2

    def teachable(self):
        _, edges, _, _, _, _ = self.G
        out = []
        for ek, u, v in edges:
            if self.edge_b.get(ek, PRIOR) >= THETA:
                continue
            if self.node_m.get(u, PRIOR) > ACCESS_FLOOR or self.node_m.get(v, PRIOR) > ACCESS_FLOOR:
                out.append((ek, u, v))
        return out

    def teach(self, ek, u, v):
        # practice succeeds in proportion to readiness, capped by learner ability
        p = min(obs_prob(self.readiness(u, v)), obs_prob(self.alpha))
        eb = self.edge_b.get(ek, PRIOR); mu = self.node_m.get(u, PRIOR); mv = self.node_m.get(v, PRIOR)
        for _ in range(ATTEMPTS):
            correct = self.rng.random() < p
            eb = bkt(eb, correct); mu = bkt(mu, correct); mv = bkt(mv, correct)
        self.edge_b[ek] = eb; self.node_m[u] = mu; self.node_m[v] = mv

    def coverage(self):
        eb = {ek: EB(m) for ek, m in self.edge_b.items()}
        perc = percolated_nodes(eb, self.G[0], theta=THETA, k_min=K_MIN)
        return len(perc) / len(self.G[2])


def choose(policy, learner, teach_list):
    _, _, _, _, betw, curv = learner.G
    if not teach_list:
        return None
    if policy == "random":
        return learner.rng.choice(teach_list)
    if policy == "easy":
        return max(teach_list, key=lambda t: learner.readiness(t[1], t[2]))
    if policy == "betweenness":
        return max(teach_list, key=lambda t: betw.get(t[0], 0.0))
    if policy == "curvature":                    # most negative shipped Forman = lowest [0,1]
        return min(teach_list, key=lambda t: curv.get(t[0], 0.5))
    if policy == "hybrid":                        # learnable bridges first
        gated = [t for t in teach_list if learner.readiness(t[1], t[2]) >= READINESS_GATE]
        pool = gated or teach_list
        return max(pool, key=lambda t: betw.get(t[0], 0.0))
    raise ValueError(policy)


POLICIES = ["random", "easy", "betweenness", "curvature", "hybrid"]


def run(n_learners=60, seed=13):
    G = load()
    nodes = G[2]
    curves = {p: [0.0] * (BUDGET + 1) for p in POLICIES}
    steps50 = {p: [] for p in POLICIES}
    steps80 = {p: [] for p in POLICIES}
    for i in range(n_learners):
        alpha = random.Random(seed + i).uniform(0.5, 0.95)
        for p in POLICIES:
            rng = random.Random((seed + i) * 97)   # paired: identical seeds/stream, only policy differs
            L = Learner(G, rng, alpha)
            cov = L.coverage(); curves[p][0] += cov
            hit50 = hit80 = None
            for step in range(1, BUDGET + 1):
                tl = L.teachable()
                choice = choose(p, L, tl)
                if choice:
                    L.teach(*choice)
                cov = L.coverage()
                curves[p][step] += cov
                if hit50 is None and cov >= 0.5: hit50 = step
                if hit80 is None and cov >= 0.8: hit80 = step
            steps50[p].append(hit50 if hit50 else BUDGET + 1)
            steps80[p].append(hit80 if hit80 else BUDGET + 1)
    res = {}
    for p in POLICIES:
        mean_curve = [c / n_learners for c in curves[p]]
        aupc = sum(mean_curve) / len(mean_curve)     # area under percolation curve [0,1]
        res[p] = {
            "AUPC": round(aupc, 4),
            "final_coverage": round(mean_curve[-1], 4),
            "median_steps_to_50pct": statistics.median(steps50[p]),
            "median_steps_to_80pct": statistics.median(steps80[p]),
            "curve": [round(x, 4) for x in mean_curve],
        }
    return res


def main():
    res = run()
    print(f"\n{'policy':>12}  {'AUPC':>6}  {'final':>6}  {'→50%':>6}  {'→80%':>6}")
    print("-" * 46)
    base = res["easy"]["AUPC"]
    for p in POLICIES:
        r = res[p]
        print(f"{p:>12}  {r['AUPC']:>6.3f}  {r['final_coverage']:>6.3f}  "
              f"{r['median_steps_to_50pct']:>6}  {r['median_steps_to_80pct']:>6}")
    win = max(POLICIES, key=lambda p: res[p]["AUPC"])
    print(f"\nfastest (max AUPC): {win}  "
          f"(+{res[win]['AUPC'] - base:+.3f} AUPC vs easy-first)")
    out = Path(__file__).resolve().parent / "curvature_sequencing_results.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"random": "#c7c7cf", "easy": "#8a8a97", "betweenness": "#0f9d6b",
                  "curvature": "#c98a12", "hybrid": "#4f46e5"}
        plt.figure(figsize=(7.2, 4.4))
        for p in POLICIES:
            plt.plot(range(BUDGET + 1), res[p]["curve"], label=p, color=colors[p],
                     lw=2.2 if p in ("hybrid", "betweenness") else 1.5)
        plt.xlabel("edges taught"); plt.ylabel("fraction of nodes percolated")
        plt.title("Sequencing to percolation: bridges vs easy-first")
        plt.legend(fontsize=9); plt.grid(alpha=.25); plt.tight_layout()
        pth = Path(__file__).resolve().parent / "curvature_sequencing.png"
        plt.savefig(pth, dpi=140); print(f"wrote {pth}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
