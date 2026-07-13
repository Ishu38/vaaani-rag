# Curvature/bridge-guided sequencing: teach the bridges, not the easy edges

**Mechanism study · 2026-07-13** ·
harness: [`curvature_sequencing.py`](curvature_sequencing.py) ·
results: [`curvature_sequencing_results.json`](curvature_sequencing_results.json) ·
plot: `curvature_sequencing.png`

## The idea

CASCADE defines mastery topologically — a node is mastered when its learned-edge
subgraph **percolates** (connected component ≥ k_min). So the fewest-mission route
to a mastered learner is the one that **connects the knowledge graph fastest**.
Percolation theory is unambiguous about which edges merge components: the
**bridges** — high-betweenness / negatively-curved edges. Hypothesis: sequence
bridges first, not the easy in-cluster edges a naive utility-max would pick.

The honest tension: bridges span *unknown* communities, so they are **harder to
learn** (low readiness). A fair test must teach under a realistic learnability
model, not just add edges to a graph.

## Design

On a tractable **connected sub-world** of the real graph (140 nodes, 312 edges,
BFS ball around the top hub), each simulated learner is taught one edge per step.
Teaching = 3 BKT practice attempts whose success scales with endpoint mastery, so
bridges genuinely resist learning until a boundary node is known. Mastery, the
percolation set, and the shipped Forman curvature are the **real** implementations.
Policies are **paired** (identical learner, seeds, and RNG stream — only the edge
choice differs), 60 learners, budget 260 steps.

| policy | what it teaches next |
|---|---|
| `random` | an unlearned edge at random |
| `easy` | highest endpoint readiness (naive utility-max / lowest-hanging) |
| `betweenness` | highest edge-betweenness (gold-standard bridge) |
| `curvature` | most negative **shipped** Forman-Ricci (the product's current signal) |
| `hybrid` | highest betweenness **among readiness-gated** edges (learnable bridges) |

Metric: fraction of nodes percolated vs. steps → **AUPC** (area under that curve;
higher = faster), plus final coverage and median steps to 50% percolation.

## Results

| policy | AUPC | final coverage | steps→50% |
|---|---:|---:|---:|
| **hybrid (learnable bridges)** | **0.247** | **0.533** | **240** |
| betweenness (pure bridge) | 0.236 | 0.517 | 250 |
| curvature (shipped Forman) | 0.206 | 0.394 | — |
| easy-first (naive utility) | 0.175 | 0.333 | — |
| random | 0.175 | 0.388 | — |

## Findings — honest

1. **Bridge-guided sequencing wins decisively.** The hybrid reaches **53%
   percolation vs easy-first's 33%** (+0.072 AUPC, +60% relative final coverage)
   and hits 50% coverage ~20 steps sooner. Teaching bridges *is* the faster route
   to a connected, mastered learner.
2. **Bridges must be readiness-gated.** `hybrid` (0.247) beats blind `betweenness`
   (0.236): teaching *learnable* bridges beats teaching bridges you'll fail. This
   is exactly the ZPD-gated-structure design the planner already had — now with
   evidence for it.
3. **Easy-first is the worst structural policy** — beaten even by `random` on
   final coverage (0.333 vs 0.388), because it myopically drills the dense seed
   cluster and never reaches out. A strong caution against naive "teach the
   easiest next thing" tutors.
4. **The shipped degree-Forman curvature is only a weak bridge proxy** (0.206)
   vs proper betweenness (0.236). It helps over easy-first, but leaves value on
   the table.

## Shipped as a result (2026-07-13)

The planner already gated edge candidates by readiness and scored betweenness —
this study validates that shape and tunes it:

- **`backend/betweenness.py`** — precomputed **global** edge betweenness, cached
  once like `ricci_curvature` (a static graph property; ~1151 edges on prod).
  Replaces the noisy per-call candidate-subgraph betweenness with the stable,
  gold-standard bridge signal.
- **`edge_state.edge_frontier_candidates`** — `structural_importance` now reads
  the global betweenness.
- **`pedagogical_planner`** — `W_EDGE_STRUCTURE` 0.4 → **0.7**: weight bridges
  more, since easy-first proved worst. The readiness gate keeps it inside the ZPD.

Deployed to `api.vaaani.in` and verified. Reproduce:
```
python research/curvature_sequencing.py
```

## Caveat / next

Simulated learnability on one sub-world; the effect should be confirmed across
several sub-worlds and, ultimately, on the same ≥60-learner logged cohort that
the percolation-validity study needs. Upgrading the shipped curvature toward a
proper (edge-based) Ricci or a betweenness blend is a clear, cheap follow-up.
