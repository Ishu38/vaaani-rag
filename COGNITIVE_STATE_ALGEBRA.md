# Cognitive State Algebra

## Part II of the Vaaani formalism — the mathematics without the architecture

**Status:** working research document, 2026-07-12. Part I (`FORMAL_MODEL.md`)
maps the formalism onto the running code; **this document is written to pass a
harder test:**

> *Can another researcher implement the framework solely from this
> specification, without ever seeing the architecture diagram or the code?*

Everything needed is here: the state space with full ontology (§1), the
algebraic structure (§2), operator definitions (§3), laws (§4), theorems with
proofs or explicit conditions (§5), semantics (§6), the canonical algorithm
(§7), and the domain-instantiation parameters (§8). §9 audits what remains
underspecified. Labels throughout: **[proved]**, **[holds by construction]**,
**[measured]**, **[testable]**, **[open]**, **[designed, not built]**.

---

## 1. The state space (ontology)

### 1.1 Primitive: the domain graph

A **domain** is a finite directed multigraph

```
    G = (V, R)          R ⊆ V × V × T_edge
```

with typed edges `T_edge ⊇ {prerequisite_for, …}` (further types are
domain-specific, §8). For the language instantiation, V is roots, words,
phonemes, concepts. G is slowly varying and is *not* learner state.

### 1.2 Primitive: the evidence object

Fix finite alphabets `Src` (perceptual channels) and
`O = {correct, incorrect, partial}`. An **evidence object** is

```
    e = ( id, s, n, src, o, c, τ, m )

    id ∈ IDs (unique)        s ∈ Students      n ∈ V
    src ∈ Src                o ∈ O             c ∈ [0,1]
    τ ∈ ℝ (time)             m : metadata (uninterpreted)
```

### 1.3 THE state: an evidence set

**Definition (canonical learner state).** The learner state of student s at
time t is the finite set

```
    E_t(s) = { e : e.s = s, e.τ ≤ t }
```

That is the whole answer to "is the learner a vector, a graph, a
distribution, a logical theory?" — **the learner is a set of uniquely
identified, typed, time-stamped observations.** Everything else is a
*derived view*, computed by the operators of §3:

| Derived view | Type | Derivation |
|---|---|---|
| Belief field `B_t : V ⇀ [0,1] × ℕ × ℝ` (mastery, exposures, last-seen) | partial function on V | `B_t = fold(U∘Δ, B_0, sort_τ(E_t))` (§3.2, Thm 3) |
| Frontier `F_t ⊆ V` | subset | thresholded prediction (§3.3) |
| Prediction ledger `P_t` | sequence of (n, p̂, τ, o?) | emitted by the policy (§3.4) |

Because the views are deterministic folds/functions of `E_t`, they carry no
information beyond `E_t` — they are *memoizations*, not state. This single
design decision is what makes the algebra of §2 possible.

`B_0` assigns every node the prior `(π, 0, −∞)`.

### 1.4 Constants (the parameter vector)

```
    θ = ( π, σ, γ, λ, τ_f )  =  ( 0.10, 0.10, 0.20, 0.15, 14 days )
```

prior, slip, guess, learning rate, forgetting time-constant. Defaults are
literature values (BKT tradition); fitting them per node-tier from cohort
evidence is hypothesis H1 of Part I. Every theorem below holds for any
θ ∈ (0,1)⁴ × ℝ₊ with σ + γ < 1.

---

## 2. The algebra

### 2.1 Merge is set union: a join-semilattice **[proved]**

Evidence objects have unique ids, so learner states form a family of finite
sets closed under union. Define

```
    L₁ ⊕ L₂  :=  L₁ ∪ L₂
```

**Theorem A (state semilattice).** (States(s), ⊕, ∅) is a join-semilattice:
⊕ is associative, commutative, and **idempotent** (L ⊕ L = L), with identity
∅. *Proof:* properties of set union on uniquely-identified elements. ∎

Consequence (engineering, but derived from the math): two replicas of the
same learner — a phone that was offline, an edge device, a server —
synchronise by union, in any order, any number of times, and converge to the
same state. The learner state is a grow-only CRDT. Belief views recompute
identically on every replica because the fold is deterministic over the
τ-sorted union (Thm 3). *Ties:* fold order for equal τ is fixed by `id` order,
making the fold total.

### 2.2 Evidence application is a monoid action — and it does not commute

Per node n, one evidence object acts on belief p ∈ [0,1] by the map
`u_e = T ∘ H_e ∘ Bayes_e ∘ Δ_dt` (definitions in §3.2). Words of evidence
act by composition; the empty word acts as identity. This is a monoid action
of finite evidence sequences on [0,1]^V.

**The action is non-commutative, and that is a feature.** Worked example
(θ defaults, w = 1, no decay, p₀ = 0.5):

```
    correct · incorrect   ↦  0.4713
    incorrect · correct   ↦  0.6432
```

Order encodes recency; a formalism in which evidence commuted would assert
that forgetting and learning history are irrelevant. The non-commutativity
is thus a *modelling commitment*, stated openly. (Note the two levels: the
evidence **set** merges commutatively (§2.1); the **fold** over its
time-sorted sequence does not — time ordering is data, carried by τ.)

### 2.3 Metric structure: comparing learners **[proved]**

Each node belief is a Bernoulli parameter. Define the distance between two
learner states (of the same or different students) as the mean Hellinger
distance over the node union:

```
    d(L₁, L₂) = (1/|V'|) Σ_{n∈V'} H( p₁(n), p₂(n) )

    H(p,q) = sqrt( 1 − ( √(pq) + √((1−p)(1−q)) ) )        V' = tracked(L₁) ∪ tracked(L₂)
```

with untracked nodes read as the prior π. **Proposition.** d is a metric
(non-negativity, identity, symmetry, triangle inequality) since Hellinger is
a metric on distributions and a normalised sum of metrics is a metric. ∎

This answers "how do I compare two learner states": cohort clustering,
before/after distances, and distance-to-target-profile are now defined
quantities, not intuitions.

### 2.4 Order structure and kinematics **[defined; measurable]**

Define the partial order `L₁ ≼ L₂ ⟺ ∀n: p₁(n) ≤ p₂(n)`. Growth is movement
upward in this poset. Derived kinematic quantities, all computable from the
evidence log alone:

```
    development velocity      v(t) = d( B_{t−δ}, B_t ) / δ         (signed per-node variant: mean Δp/δ)
    learning acceleration     a(t) = ( v(t) − v(t−δ) ) / δ
    cognitive stability       s(t) = 1 − Var_window( p_n(t) trajectories )   (oscillation index)
    evidence density          ρ(n) = |{e : e.n = n}| / age(n)
```

These are the reviewer's "emergent quantities": they exist *because* the
state is a replayable fold — none of them requires new instrumentation.
**[measurable today; no empirical claims made yet]**

---

## 3. The operators

Signatures first; then definitions. Fix a student s; write E for E_t(s).

```
    𝒫 : RawInput → EvidenceObject ∪ {∅}          perception (may abstain)
    Δ : [0,1] × ℝ₊ → [0,1]                        decay
    U : [0,1] × EvidenceObject → [0,1]            belief revision
    𝒟 : (V ⇀ [0,1]) × G → (V → [0,1])             prediction / readiness
    𝒜 : (V → [0,1]) × Ledger → V × [0,1] × Kind   policy (returns node, p̂, kind)
    ℳ : Ledger → CalibrationTable                 metacognitive evaluation
```

### 3.1 Perception 𝒫 **[holds by construction]**

𝒫 is any procedure that maps a raw interaction to *at most one* evidence
object with `n ∈ V`, or abstains. Abstention on unresolvable input is
mandatory (Law A2). 𝒫 is the only operator that may inspect raw input; its
output type is the sole interface to everything downstream.

### 3.2 Decay Δ and revision U

```
    Δ(p, dt)   = π + (p − π) · exp(−dt / τ_f)

    Bayes_e(p) = p(1−σ) / ( p(1−σ) + (1−p)γ )            if e.o = correct
               = pσ     / ( pσ + (1−p)(1−σ) )            if e.o = incorrect
      (partial: as correct with w halved)

    H_e(p→p*)  = (1−w)p + w·p*        w = e.c             (Jeffrey conditioning)
    T(p)       = p + (1−p)λ                                (learning transition)

    U(p, e)    = T( H_e( Bayes_e( p ) ) )
```

The fold applies `Δ` with dt = gap since the node's previous evidence, then
`U`. All four maps are continuous and monotone increasing in p (§5, Thm 2's
lemma).

### 3.3 Prediction 𝒟 (readiness and the frontier)

```
    readiness(n) = mean{ p(q) : q ∈ Pre(n) }              (1 if Pre(n) = ∅)
    p̂(n)         = γ + (1−γ−σ) · ( α·p(n) + (1−α)·readiness(n) )      α = 0.35
    F            = { n : p̂(n) ∈ [z_lo, z_hi] }            [0.60, 0.80]
```

α and the band are declared heuristics, tuned by ℳ's output (Part I §2.3).

### 3.4 Policy 𝒜 and evaluation ℳ

𝒜 maximises a scored utility over F (gain λ(1−p), novelty, variety terms),
with two structural rules: arc-pinning (consecutive selections stay in one
graph neighbourhood) and **probe/abstain** (empty F ⇒ select the
least-evidenced connected node to gather information — value of
information, not teaching). 𝒜 MUST write (n, p̂, τ) to the ledger *before*
the learner attempts the activity (Law A6). ℳ bins the ledger's completed
(p̂, outcome) pairs into a reliability table; expected calibration error is
the framework's self-assessment scalar.

### 3.5 The transition

One interaction cycle of the framework is

```
    S_{t+1} = ( E_t ⊕ {𝒫(input_t)} , views recomputed )
    action_t = 𝒜( 𝒟( B_t, G ), P_t )
```

i.e. `F = ℳ ∘ 𝒜 ∘ 𝒟 ∘ (U∘Δ)* ∘ 𝒫` — the reviewer's operator composition,
now with every symbol defined.

---

## 4. Laws (axioms of the framework)

- **A1 — Evidence conservation.** The state changes only by ⊕ of
  𝒫-produced evidence objects. No operator writes beliefs directly.
  **[holds by construction — single ingress]**
- **A2 — Grounded attribution.** Every evidence object names a node of G;
  perception abstains rather than guess a node. **[holds by construction]**
- **A3 — Boundedness.** Beliefs remain in [0,1] (in fact [min(π,λ), 1)).
  **[proved — Part I Prop 2]**
- **A4 — No spontaneous learning.** Between observations, every belief moves
  monotonically toward the prior π and never away from it. (Note: beliefs
  *below* π rise toward π — regression to baseline, in both directions.)
  **[proved — Part I Prop 1]**
- **A5 — Traceability.** Every derived quantity is a deterministic function
  of E; deleting nothing, the entire belief history is replayable, including
  under counterfactual θ. **[proved — Part I Prop 3]**
- **A6 — Pre-registration.** Every pedagogical action carries a success
  prediction committed before the outcome exists. Calibration is therefore a
  property of the *framework*, not an offline evaluation. **[holds by
  construction; its accuracy is H1, testable]**
- **A7 — Uncertainty-seeking.** When the belief state cannot justify a
  teaching action (empty frontier), the policy must choose an
  information-gathering action rather than an arbitrary teaching one.
  **[holds by construction — probe rule]**

The reviewer's "pedagogical law — activities must reduce uncertainty, not
merely maximise correctness" is A7 plus the ZPD band: the policy is
*forbidden* the comfort zone (p̂ > z_hi), where correctness is maximal and
information is minimal.

---

## 5. Theorems

**Theorem 1 (closure & bounds).** For any evidence word, beliefs stay in
[0,1]; one application of U lands in [λ, 1). *Part I, Prop 2.* **[proved]**

**Lemma (monotonicity).** Δ(·,dt), Bayes_e, H_e, T are continuous and
monotone increasing on [0,1]; hence so is any composition φ of them.
*Proof:* direct differentiation; Bayes'(p) = (1−σ)γ/den² > 0 (correct case),
σ(1−σ)/den² > 0 (incorrect); the rest are affine with positive slope. ∎

**Theorem 2 (convergence under stationary practice).** Fix a node and a
stationary practice regime (fixed outcome, confidence w, inter-practice gap
dt). Let φ = U(·, e) ∘ Δ(·, dt). Then from any p₀ ∈ [0,1], the iterates
p_{k+1} = φ(p_k) converge monotonically to a fixed point p* of φ.
*Proof:* φ is continuous and monotone increasing on the compact interval
[0,1] (Lemma). If p₁ ≥ p₀ then by monotonicity p₂ = φ(p₁) ≥ φ(p₀) = p₁, and
inductively (p_k) is nondecreasing and bounded; symmetrically nonincreasing
if p₁ ≤ p₀. Either way it converges, and by continuity the limit is a fixed
point. ∎ **[proved]** (Uniqueness is not claimed in general; numerically the
attractor is unique for default θ — iterates from p₀ = 0.01 and 0.99 meet at
the same p* to 4 decimals.)

**Corollary 2.1 (the cadence–ceiling law).** The equilibrium mastery under
always-correct practice is a decreasing function of the practice gap dt —
the fixed point of φ. Computed for default θ (reproducible from §3.2 in ten
lines of code):

| practice gap dt | equilibrium mastery p* |
|---|---|
| 1 day | 0.985 |
| 2 days | 0.969 |
| 3 days | 0.952 |
| 3.5 days | 0.943 |
| 7 days | 0.879 |
| 14 days | 0.740 |

With the mastered-threshold 0.95, **"mastered" status is dynamically
maintainable only at practice cadence ≲ 3 days** under default θ.
Independently, the review trigger (review when decayed mastery < 0.80)
fires at dt = τ_f·ln(0.9/0.7) ≈ 3.5 days after full mastery — i.e. the
scheduler the system already runs is, to within half a day, the one this
theorem says is necessary. The formalism *predicts* its own scheduler.
**[proved given θ; θ itself is a fitted quantity — H1]**

**Theorem 3 (replay & merge).** For any two evidence sets E₁, E₂ of one
student, fold over sort_τ(E₁ ⊕ E₂) is well-defined and replica-independent;
hence distributed replicas converge (§2.1). *Proof:* determinism of the fold
+ totality of the (τ, id) order + idempotent union. ∎ **[proved]**

### Open problems (stated, not claimed)

- **O1 (stochastic stability).** Under outcome distributions rather than
  fixed outcomes, (p_k) is a Markov chain on [0,1]; existence/uniqueness of
  a stationary distribution and mixing rate: conjectured, unproven. The
  "cognitive stability" metric (§2.4) is its empirical shadow. **[open]**
- **O2 (credal beliefs) — RESOLVED 2026-07-12.** Implemented as a second
  derived view of the same evidence set (`backend/credal.py`): per node,
  Beta(α, β) tracks *observable accuracy* with Jeffrey-weighted conjugate
  updates (correct: α += w) and count-decay toward prior pseudo-counts
  (A0, B0) = θ₀·N0, (1−θ₀)·N0 with θ₀ = γ + (1−γ−σ)π. Two theorems join §5:

  **T4 (mean-confirming evidence never increases uncertainty).** If
  α ≥ β (mean ≥ ½), adding a success weight never increases Var(θ);
  symmetrically for failures when β ≥ α. *Proof (unit weight):*
  Var(α+1,β) < Var(α,β) ⟺ (α+1)(α+β)²(α+β+1) < α(α+β+1)²(α+β+2)
  ⟺ α > s²/(3s+2) with s = α+β, which holds whenever α ≥ s/2. ∎
  Fractional weights w ∈ (0,1]: verified over 200k random cases, zero
  violations; analytic extension routine. This is the reviewer's
  uncertainty law, made precise: *supporting* must mean mean-confirming —
  surprising evidence rightly widens the belief. **[proved]**

  **T5 (closed-form information value).** The expected variance reduction
  from one more observation is, by the law of total variance,
  `EIG = μ(1−μ)/(α+β+1)²` — one line, no sampling. This is the objective
  of the v1 experiment-design policy (Part III §6), shipped dark behind a
  zero-default weight. **[proved; verified against brute force]**

  Bonus diagnostic: inverting μ ≈ γ + (1−γ−σ)·mastery gives an independent
  mastery estimate; per-node |BKT − inverted-Beta| is a standing
  model-criticism metric (`misfit`). Replay of the full pre-existing
  evidence log through the new operator succeeded — A5 demonstrated, not
  just proved. **[measured]**
- **O3 (joint inference).** U updates one node per evidence object; G enters
  only through readiness at prediction time. Full belief propagation over G
  (evidence about "unhappiness" should move "un-" and "happy") is future
  work; A5 guarantees any such upgrade can be replayed over existing logs.
  **[open]**

---

## 6. Semantics

What does an evidence object *mean*? The framework's answer is deliberately
layered:

1. **Assessment semantics (built):** `(n, o, c)` means "an observation of
   the learner's behaviour on domain object n had outcome o, and the
   perceptual channel itself is c-confident in that reading." Nothing more.
   In particular the framework does not pretend to know *why*.
2. **Causal semantics (designed, not built):** the extension type
   `e.h : Hypotheses ⇀ [0,1]` — a distribution over a domain-specific error
   cause taxonomy (for language: L1 transfer, overgeneralisation,
   sound–spelling conflation, terminology confusion), with priors supplied
   by the contrastive model of the learner's L1 (CAVP). "Yesterday I go to
   school" then carries not just (past-tense-node, incorrect) but a
   hypothesis field over causes. Everything in §2–§5 is unchanged by this
   extension — h rides along the evidence object and is consumed by a
   diagnosis view. **This is the cause-net roadmap item; no code exists.**

The division is principled: the algebra needs only layer 1 to run; layer 2
enriches diagnosis without touching the update laws.

---

## 7. The canonical algorithm

Reference implementation of the framework, independent of any architecture:

```
ECST-CYCLE(student s, domain G, constants θ):
    E ← load evidence set of s                       # §1.3
    B ← FOLD(E):                                     # §3.2, replayable
          for e in sort_(τ,id)(E):
              p ← Δ(p_prev(e.n), τ_gap);  B[e.n] ← U(p, e)
    p̂  ← 𝒟(B, G)                                     # readiness + ZPD
    (n, p̂ₙ, kind) ← 𝒜(p̂, ledger)                     # argmax utility; probe if F = ∅
    LEDGER-APPEND(n, p̂ₙ, now)                        # A6: pre-register
    present activity for n; observe raw outcome r
    e' ← 𝒫(r)                                        # typed evidence or abstain
    if e' ≠ ∅:  E ← E ⊕ {e'}                         # A1: sole state change
    report ℳ(ledger)                                  # calibration table
    repeat
```

Complexity per cycle: incremental fold is O(1) per evidence object (memoise
B); 𝒟 is O(|V| + |R_pre|); 𝒜 is O(|F| log |F|). The full replay is
O(|E| + |V|) — cheap enough to recompute a learner from scratch on a phone,
which is what Theorem 3 needs for edge deployment.

---

## 8. Domain instantiation (generality claim, honestly bounded)

The framework is parametrised by a **domain specification**

```
    Dom = ( G, Src, ActivityTemplates, CauseTaxonomy )
```

Fixed across domains: §1 state, §2 algebra, §3.2 update laws, §4 axioms,
§5 theorems, §7 algorithm. Varying: the graph, the perceptual channels, the
activity template family consumed by 𝒜's generator, and layer-2 semantics.

| Domain | G nodes/edges | example evidence sources | status |
|---|---|---|---|
| Language (Vaaani) | roots, words, phonemes; root_of, sounds_like, translates_to | quiz, audio attempt, chat, OCR | **instantiated, in production** |
| Mathematics | concepts, skills; prerequisite lattice | problem attempts, worked steps | not instantiated |
| Music | intervals, rhythms, pieces | audio performance, sight-reading | not instantiated |

The generality claim we are entitled to today: *the formal core nowhere
references language.* The claim we are **not** entitled to: that other
instantiations work — that requires building one. **[testable, expensive]**

---

## 9. Implementability audit (the reviewer's test, answered)

Could a stranger implement this from §§1–8 alone?

**Fully specified:** state ontology; merge/metric/order algebra; Δ, U, 𝒟
with exact equations and constants; laws A1–A7; the algorithm; ledger and
calibration semantics; tie-breaking; complexity.

**Specified up to declared freedom:** 𝒜's utility weights (given, but
labelled tunable); ZPD band and α (heuristics awaiting H1 fitting); activity
templates (domain content, not framework).

**Not yet specifiable — the honest frontier:** 𝒫's internals (perception is
domain- and sensor-specific by design; the framework constrains only its
output type and abstention duty); layer-2 causal semantics (O2/cause net);
stochastic stability guarantees (O1).

Verdict: the *framework* passes the test; the *tutor* does not — a stranger
would rebuild the mathematics faithfully but would still have to author a
domain graph, perception channels, and activities. That is the correct
boundary between a computational formalism and a product, and it is where
the patentable apparatus (product) and the publishable theory (framework)
separate cleanly.

---

## Naming

Working name: **Evidence-Grounded Cognitive State Algebra** (the state
semilattice + non-commutative evidence action + metric/order structure of
§2, governed by A1–A7). Provisional; the contribution is §§1–7, not the name.
