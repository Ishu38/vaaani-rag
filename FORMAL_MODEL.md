# The Learner as an Evidence-Grounded Cognitive State

## A computational formalism for the Vaaani architecture

**Status:** working research document, 2026-07-12.
**Relation to code:** every equation in §2–§5 is the *exact* rule implemented in
`backend/` (file:line cited). Nothing here is aspirational unless explicitly
marked **[designed, not built]** or **[open]**. Companion engineering document:
`COGNITIVE_ARCHITECTURE.md`. **Part II** (`COGNITIVE_STATE_ALGEBRA.md`) lifts
this into a code-independent algebra: state semilattice, metric/order
structure, laws A1–A7, convergence + cadence–ceiling theorems, canonical
algorithm, domain-instantiation parameters.

---

## 0. Thesis

> A learner is represented not as a score or an embedding, but as a
> continually evolving, evidence-grounded cognitive state that is revised
> through explicit reasoning and used to guide future interactions.

The scientific claim is **not** any single component — each is credited prior
work (§6). The claim is a *property of the composition*: the learner state is
a typed mathematical object whose every value is a deterministic function of
an inspectable evidence log, whose updates are closed-form probabilistic
rules, and whose predictions are logged *before* outcomes so the system's own
honesty is a measured quantity. LLMs appear only at the generation surface
(phrasing), never in the state, the update, or the decision.

---

## 1. The state object

The learner state at time *t* is the tuple

```
    L_t = ( G_W , E_t , B_t , P_t )
```

| Symbol | Name | Implementation |
|---|---|---|
| `G_W` | Linguistic World Model — directed graph of language objects (roots, words, phonemes, concepts) with typed edges incl. prerequisite structure | `data/graph.json` (785 nodes) + `data/prereq_overlay.json`; loaded by `development_engine.WorldModel` |
| `E_t` | Evidence corpus — the append-only log of typed observations up to *t* | `evidence` table, `backend/evidence_graph.py` |
| `B_t` | Belief field — per-node mastery beliefs with timestamps | `twin` table, `backend/cognitive_twin.py` |
| `P_t` | Prediction ledger — every P(success) the system asserted, later joined with the actual outcome | `predictions` table, `backend/cognitive_twin.py` |

`G_W` is slowly varying (curriculum ingestion); `E_t`, `B_t`, `P_t` evolve
per interaction. The state is *not* a score: no scalar summarises it. It is
*not* an embedding: every coordinate has a linguistic name and a provenance
chain back to observations.

### 1.1 The evidence type (the perception boundary)

Every perceptual subsystem — quiz answer, mission outcome, spaced review,
chat turn, OCR, audio attempt — must reduce its observation to one value of a
single type before anything downstream may consume it
(`evidence_graph.EvidenceObject`):

```
    e = ( s, n, src, o, c, τ, m )

    s   ∈ Students        n ∈ nodes(G_W)         src ∈ {quiz, mission, review,
    o   ∈ {correct, incorrect, partial}                 chat, ocr, audio, seed}
    c   ∈ [0,1]  perceptual confidence in the observation itself
    τ   ∈ ℝ      timestamp        m : metadata (free)
```

This is the formal content of "evidence-grounded": raw input is
inadmissible; only `e` values enter the state. The typed alphabet
(src × o × c) is deliberately small — it is the *interface contract* between
perception and cognition, in the spirit of evidence-centered assessment
design (Mislevy et al., §6).

The constraint `n ∈ nodes(G_W)` is enforced at every ingress point
(`cognitive_loop_routes.py` 404s unknown nodes; the quiz bridge
`simulation/evidence_bridge.py` emits nothing rather than emit to a guessed
node). Mis-attributed evidence corrupts `B_t`; silence does not.

---

## 2. The update operators

### 2.1 Temporal decay operator Δ (forgetting)

Between observations, belief relaxes toward the population prior
(`cognitive_twin._decayed`, constants at `cognitive_twin.py:35-40`):

```
    Δ_dt(p) = π + (p − π) · e^(−dt / τ_f)          π = 0.10,  τ_f = 14 days
```

Applied lazily on read, so the stored state never has to be swept.

**Proposition 1 (fixed point & contraction).** Δ_dt is an affine contraction
on [0,1] with Lipschitz constant e^(−dt/τ_f) < 1 for dt > 0 and unique fixed
point π. Hence an unobserved node's belief converges monotonically to the
prior — "the system forgets that it knew, at a known rate."
*Proof:* |Δ_dt(p) − Δ_dt(q)| = e^(−dt/τ_f)|p − q|; Δ_dt(π) = π. ∎

Spaced-review urgency is not a separate mechanism; it *falls out* of Δ
(planner review rule, §4).

### 2.2 Belief revision operator U (evidence ingestion)

One evidence object updates exactly one node belief
(`cognitive_twin.update`, `cognitive_twin.py:102-136`). With current belief
p (post-decay), slip σ = 0.10, guess γ = 0.20, learning rate λ = 0.15:

Step 1 — Bayesian posterior (two-state HMM emission update, classical BKT):

```
    o = correct:     p* = p(1−σ) / ( p(1−σ) + (1−p)γ )
    o = incorrect:   p* = pσ     / ( pσ     + (1−p)(1−σ) )
    o = partial:     treated as correct with confidence halved (w ← w/2)
```

Step 2 — soft-evidence blend (Jeffrey conditioning with weight w = c):

```
    p' = (1 − w) p + w p*
```

w = 1 is a fully trusted observation; w = 0 changes nothing. This is where
perceptual uncertainty (e.g. audio quality from the pronunciation ear) enters
the cognition *as mathematics*, not as a heuristic.

Step 3 — learning transition:

```
    p'' = p' + (1 − p') λ
```

**Proposition 2 (closure).** If p ∈ [0,1], w ∈ [0,1] then p'' ∈ [λ, 1] ⊂ [0,1].
*Proof:* p* is a ratio of nonnegative terms bounded by its denominator; p' is
a convex combination; p'' is a convex combination of p' and 1 with weight λ. ∎

**Proposition 3 (replayability / auditability).** U and Δ are deterministic,
and E_t is append-only. Therefore

```
    B_t = fold( U∘Δ , B_0 , E_t )
```

— the entire belief field is a deterministic fold of the evidence log and can
be *replayed, audited, or recomputed under different constants* at any time.
This is the formal content of "revised through explicit reasoning": no belief
exists that cannot be reconstructed from named observations. A weight-updated
neural learner model has no analogous property.

### 2.3 Readiness and the numeric ZPD (prediction operator)

For node n with prerequisite set Pre(n) ⊆ G_W
(`development_engine.p_success`, `development_engine.py:78-87`):

```
    readiness(n) = mean{ mastery(q) : q ∈ Pre(n) }        (1 if Pre(n) = ∅)
    P̂(success | n) = γ + (1 − γ − σ) · ( 0.35·mastery(n) + 0.65·readiness(n) )
```

The Zone of Proximal Development is made numeric: the frontier is

```
    F_t = { n : P̂(success|n) ∈ [0.60, 0.80] }
```

Below the band: frustration zone. Above: comfort zone, no growth. **The
0.35/0.65 blend and the band are v0 heuristics** — declared as such, and the
calibration ledger (§5) is the instrument that will tune them against
reality. (Vygotsky supplies the construct; the operationalisation is ours.)

---

## 3. The policy (decision-theoretic, no RL, no LLM)

`pedagogical_planner.select_activity` (`pedagogical_planner.py:80-173`) is
myopic expected-utility maximisation over the frontier:

```
    score(n) = 1.0 · λ(1 − mastery(n))         expected mastery gain (BKT lookahead)
             + 0.2 · 1[exposures(n) = 0]       novelty
             + variety(n)                      tier rotation / anti-repeat pressure
    review:  score = 0.6 · (0.80 − mastery)    urgency, for decayed once-learned nodes
```

with two structural rules layered on argmax:

- **Expedition pinning** — an active arc pins selection to its queue so
  consecutive missions stay inside one neighbourhood of `G_W` (coherent
  journey rather than i.i.d. sampling).
- **Probe/abstain** — if no candidate exists, the belief state is too thin
  to plan; the agent selects the least-evidenced structurally-connected node
  *explicitly to learn about the learner*. This is a value-of-information
  move (Howard 1966; AIMA ch. 16): acting to reduce the agent's own
  uncertainty rather than guessing.

**The metacognitive commitment:** every decision writes
`(student, node, P̂(success))` to the prediction ledger *before* the learner
attempts the activity (`twin.log_prediction`). The system goes on record.

---

## 4. Closing the loop

The full interaction cycle, entirely in the operators above:

```
    observe → e (typed evidence) → B ← U∘Δ (belief revision)
        → F (frontier prediction) → π (decision + logged P̂) → activity
        → observe …
```

One transition of the formalism is therefore

```
    L_{t+1} = F( L_t , o_t )
```

where F is the composition of §2–§3 — a specified state-transition operator,
not a workflow description. Activities themselves are generated symbolically
from `G_W` edges with answer keys and deterministic distractors
(`activity_generator.py`); the LLM, where present at all, phrases content and
never chooses it.

---

## 5. Measured properties (what the composition buys)

Each candidate "emergent property" is stated with its current evidential
status. Discipline: **measured ≠ predicted ≠ designed.**

### 5.1 Calibration honesty — instrument EXISTS, data accruing

Because P_t logs predictions before outcomes, the reliability table

```
    calibration bin b:  ( mean P̂ , empirical success rate , n )
```

is computed directly from the ledger (`cognitive_twin.calibration`,
`/loop/calibration/{sid}` live in production). This yields expected
calibration error (ECE) as a *standing, per-learner, always-on metric* — the
system's confidence is an auditable quantity, not a claim. **[measured
instrument; needs cohort volume before any accuracy claim]**

### 5.2 Traceable justification — property HOLDS by construction

Every mission carries a human-readable reason naming the quantities that
produced it (P̂, expected gain, readiness, rotation). Proportion of decisions
with traceable justification is 100% by construction — the interesting
empirical question is *agreement with expert teacher judgment*, which is an
open study (§7, H3). **[structural property; expert-agreement unmeasured]**

### 5.3 Sample-efficient adaptation — OPEN

Define adaptation efficiency η = Δ(learning gain) / interactions. The
hypothesis: structured evidence + graph readiness personalises in fewer
interactions than an unstructured baseline. **[open; requires the cohort
study in §7 — no learning-gain claims are made today]**

### 5.4 Developmental forecasting — PARTIAL mechanism, OPEN validation

The state already predicts P̂_{t} (next-attempt success). Forecasting
P_{t+k} (trajectory over k future steps) follows from iterating U∘Δ under a
policy — the mechanism is closed-form, but its accuracy against longitudinal
data is unmeasured. **[mechanism exists; validation open]**

### 5.5 Causal diagnosis — DESIGNED, NOT BUILT

Moving from "wrong" to "wrong *because* L1 phonology merges /v/–/w/" requires
the cause Bayesian network with CAVP L1 priors. Explicitly not implemented;
do not claim it. **[designed, not built]**

---

## 6. Position among prior formalisms (credit where due)

The composition is the contribution; the parts are not:

- **Bayesian Knowledge Tracing** — Corbett & Anderson (1994). §2.2 steps 1,3
  are classical BKT with literature-default parameters.
- **Jeffrey conditioning / soft evidence** — Jeffrey (1965); AIMA ch. 14’s
  uncertain-evidence treatment. §2.2 step 2.
- **Forgetting as exponential relaxation** — Ebbinghaus tradition; ACT-R
  activation decay (Anderson). §2.1.
- **Knowledge space theory** — Doignon & Falmagne (1985–99); ALEKS. The
  prerequisite-driven frontier is kin to their outer fringe, computed here on
  a *linguistics-native* graph with probabilistic (not set-theoretic) state.
- **Evidence-centered assessment design** — Mislevy, Steinberg & Almond. The
  EvidenceObject boundary is ECD's evidence-model discipline made an API type.
- **Open learner models** — Bull & Kay. Reconstructibility (Prop. 3) is an
  OLM guarantee at the *architecture* level rather than a UI feature.
- **Teaching as decision/POMDP planning** — Rafferty et al. (2016); VOI —
  Howard (1966). §3 is the myopic special case, chosen deliberately for
  explainability and edge-device budgets.
- **ZPD** — Vygotsky; numeric operationalisation is ours and flagged v0.

**Claimed novelty (to be defended, not assumed):** (i) the typed
evidence-object boundary as the *sole* percept-cognition interface in a
deployed tutor; (ii) per-decision pre-registered predictions making
calibration a standing product metric; (iii) belief replayability as an
architectural invariant; (iv) the linguistics-native graph (roots, phonemes,
cross-script cognates) with L1-transfer priors as the substrate — (iv) jointly
with CAVP is where the field-specific contribution lies.

---

## 7. Falsifiable hypotheses and the study design

- **H1 (calibration):** after parameter fitting on cohort evidence, per-bin
  |P̂ − empirical| ≤ 0.10 (ECE ≤ 0.05) on held-out interactions. *Instrument
  live; test = fit-then-freeze, evaluate on later weeks.*
- **H2 (sample efficiency):** η(full state) > η(ablated) with readiness
  forced to 1 (graph off) and with w forced to 1 (soft evidence off).
  *Within-system ablations are the honest first study — no external cohort
  needed; the ablation only changes §2–§3 constants/terms.*
- **H3 (diagnostic agreement):** system-generated reasons agree with blinded
  expert teacher judgments above chance (Cohen's κ > 0.4) on the same
  evidence windows. *Needs 2 raters + ~100 decision instances.*
- **H4 (forecasting):** iterated U∘Δ predicts week-ahead per-node success
  better than last-observation-carried-forward and better than a global
  logistic baseline. *Longitudinal; runs on the same logs.*

Baselines that must be beaten honestly: plain BKT without graph readiness;
an LLM-only tutor with no state; random-within-curriculum policy. Metrics:
ECE, η, κ, Brier score for H4.

**Sequencing** (matches the product's constraint that all instruments are
already logging in production): (1) accumulate cohort evidence through the
school pilot; (2) fit BKT parameters per node tier (currently
literature-default — known limit); (3) run H1/H4 on logs, H2 as ablation,
H3 with examiners; (4) only then any learning-gain claim.

Publication route, in order of readiness: AIED/EDM short paper on the
formalism + calibration instrument (§1–§5 are complete and running); full
paper once H1–H2 land; IJAIED for the longitudinal study.

---

## 8. Known limits (standing honesty list)

1. BKT parameters (σ, γ, λ, π, τ_f) are literature defaults, not fitted.
2. The 0.35/0.65 mastery/readiness blend and the [0.60, 0.80] band are
   heuristics awaiting calibration-driven tuning.
3. Per-node independence: U updates one node per evidence object; cross-node
   dependence enters only through readiness at prediction time. Joint
   inference over `G_W` (belief propagation) is future work.
4. The cause Bayes net (causal diagnosis) is designed, not built — §5.5.
5. Prerequisite coverage of `G_W` is partial (485/785 nodes lack curated
   prereqs; overlay derivation is heuristic).
6. No learning-gain, no efficacy claims exist today. The system measures its
   own calibration; that is all it can currently assert about itself.
