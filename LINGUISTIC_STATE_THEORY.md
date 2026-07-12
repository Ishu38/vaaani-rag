# Λ — The Linguistic Cognitive State

## Part III of the Vaaani formalism: a computational theory of linguistic development

**Status:** working research document, 2026-07-12. Part I (`FORMAL_MODEL.md`)
grounds the mathematics in running code; Part II
(`COGNITIVE_STATE_ALGEBRA.md`) is the domain-independent algebra. **Part III
commits the theory to its home discipline.** The framework is no longer
pitched as an AI-tutoring architecture but as:

> **A computational theory of second-language development, in which the
> learner is modelled as an evidence-grounded linguistic cognitive state
> Λ_t, revised by explicit operators, whose evolution is the object of
> study — and the tutor is one experimental apparatus for testing it.**

The slogan the system lives by: **it computes belief revisions, not
answers.** Labels as before: [proved] / [holds by construction] /
[measured] / [testable] / [designed, not built] / [open].

---

## 1. The central object

Part II established that the canonical learner state is the evidence set
E_t, with the belief field B_t : V ⇀ [0,1] as its principal derived view.
Part III adds the *linguistic typing* that turns B_t from an AI object into
a linguistics object.

### 1.1 Tier-typed domain graph

V is partitioned by linguistic tier, **derived from node and edge types
already present in the data** (node `type` ∈ {phoneme, grapheme, root,
prefix, word, meaning, …}; edge types sounds_like, written_as, root_of,
word_family, translates_to, cognate_with, means):

```
    V = V_Φ ⊎ V_Μ ⊎ V_Λex ⊎ V_Χ ⊎ V_⊥
```

Measured on the production graph (785 nodes, 2026-07-12):

| Tier | Symbol | Content | Nodes today |
|---|---|---|---|
| Phonology–orthography | V_Φ | phonemes, graphemes, sound–spelling links | 264 |
| Morphology | V_Μ | roots, affixes, word families | 239 |
| Cross-linguistic | V_Χ | translation/cognate bridges to the learner's L1 | 168 |
| Lexical–semantic | V_Λex | words, meanings | 45 |
| Untyped residue | V_⊥ | curation debt | 69 |
| **Syntax** | V_Σ | — | **0** |
| **Pragmatics/discourse** | V_Ρ | — | **0** |

The zeros are stated, not hidden: the current instantiation is a theory of
*lexical, morphological, phonological and cross-linguistic* development.
Syntax and pragmatics are tier slots the formalism already accommodates
(nothing in Parts I–II changes) but for which no domain content exists.
V_Χ deserves emphasis: a **cross-linguistic tier is not standard in learner
models** — it exists here because the system was built L1-first (Bengali/
Hindi), and it is where CAVP's contrastive priors attach.

### 1.2 The Linguistic Cognitive State

```
    Λ_t = ( B_t|Φ , B_t|Μ , B_t|Λex , B_t|Χ ; D_t , U_t , H_t )
```

- **B_t|τ** — the belief field restricted to tier τ: the learner's
  *interlanguage profile* by linguistic level. **[measured — derived view,
  computable today; see scripts/lambda_state.py]**
- **D_t — developmental stage.** Defined abstractly: fix a finite stage
  lattice (Σ_stages, ⊑) and, for each stage, a *criterial set* C_σ ⊆ V of
  structures; D_t = the ⊑-maximal σ with mean mastery over C_σ ≥ θ_σ.
  The definition is formal now; **the stage inventory is domain content**
  (for SLA the natural source is Processability Theory's implicational
  stages). **[defined; stage inventory not yet authored]**
- **U_t — quantified uncertainty.** Per-node Beta(α, β) over observable
  accuracy, Jeffrey-weighted, count-decaying (uncertainty grows with
  disuse). Built 2026-07-12 (`backend/credal.py`; Part II O2 → theorems
  T4/T5): the mean-confirming-evidence law is proved, expected information
  gain is closed-form, and the BKT-vs-Beta misfit is a standing
  model-criticism diagnostic. Live views: `/loop/credal/{sid}` and the ±sd
  column of `scripts/lambda_state.py`. **[measured]**
- **H_t — competing linguistic hypotheses.** The hypothesis field of
  Part II §6 layer 2, now stated in SLA terms (§3). **[designed, not built]**

Nothing else about the learner exists in the theory. Every component above
is (or will be) a deterministic view of the evidence set — Part II's
replayability (A5) extends to Λ_t as a whole.

---

## 2. Evidence, linguistically typed

Part II's evidence object is unchanged; the *linguistic reading* of
(n, o, c) is fixed by n's tier:

| Tier of n | An evidence object means… | Perceptual channel today |
|---|---|---|
| V_Φ | a sound/spelling discrimination or production observation (e.g. /f/ realised as /p/) | quiz; **pronunciation ear** (wav2vec2 forced alignment → per-phone outcomes with acoustic confidence — LIVE in production 2026-07-12: `vaaani-ear` service on vaaani-vm, SPA mic on /learn, per-phone evidence verified end-to-end) |
| V_Μ | a word-building / affix observation ("walked → walk") | missions, quiz |
| V_Λex | word–meaning association | missions, chat, quiz |
| V_Χ | an L1-bridge observation (cognate recognised, false friend confused) | bridge missions |

The pronunciation channel is the showcase of the confidence semantics: its
c comes from measured audio quality, so Jeffrey conditioning (Part II §3.2)
weighs a noisy recording *mathematically less* than a clean one. No other
part of the framework needs to know why.

---

## 3. Linguistic inference: from "wrong" to hypotheses **[designed, not built]**

The reviewer's example, formalised. Observation: learner produces
*"Yesterday I go."* A grading system emits (past-tense node, incorrect).
Linguistic inference instead emits an evidence object carrying a
**hypothesis field**

```
    h : { H_A: past-tense morphology unacquired,
          H_B: temporal-adverb cue ignored,
          H_C: L1 transfer (Bengali/Hindi tense–aspect mapping),
          H_D: performance slip } → [0,1]
```

with priors P(H | L1) supplied by the contrastive model (CAVP) — this is
where fifteen years of L1-interference work becomes a *parameter of the
theory* rather than product colour. Diagnosis is then belief revision over
causes, exactly parallel to belief revision over mastery. The update law
for h (a Bayes net over cause taxonomies per tier) is the single largest
unbuilt component of the theory; its type and its position in Λ_t are now
fixed, so building it changes no existing mathematics. Requires V_Σ content
for tense/agreement phenomena — the two gaps are one gap.

---

## 4. Acquisition dynamics

The theory's dynamical claim is Part II's transition, re-read:

```
    Λ_{t+1} = 𝒯( Λ_t , E_t )                      deterministic given evidence
    P( Λ_{t+1} | Λ_t , policy )                    stochastic over learner outcomes [open — O1]
```

with the proved structure carried over: bounded beliefs, monotone
convergence under stationary practice, and the **cadence–ceiling law**
(Part II, Cor. 2.1): equilibrium mastery is a decreasing function of
practice interval, and "mastered" is dynamically maintainable only at
cadence ≲ 3 days under default parameters — i.e. *maintenance is a property
of the interaction regime, not of the learner alone.* In SLA terms this is
a computable model of attrition-under-disuse.

**Positioning within SLA theory.** Dynamic Systems Theory approaches to SLA
(Larsen-Freeman; de Bot, Lowie & Verspoor) argue that language development
is a self-organising dynamical system — but the DST-SLA literature is
overwhelmingly qualitative. Λ_t is offered as a *computable* DST
instantiation: an explicit state space, explicit transition operators, and
theorems in place of metaphors. Interlanguage (Selinker) supplies the
reading of B_t as a systematic learner variety; Processability Theory
(Pienemann) supplies candidate stage inventories for D_t; the ZPD
(Vygotsky) is operationalised as the frontier band; the guess/slip/transit
core is Corbett & Anderson's BKT. What the theory adds over learner-model
tradition: the object being tracked is the learner's *linguistic
representation by tier including the L1 bridge*, not task-level skill —
and every value of it is replayable from typed evidence.

---

## 5. What is conserved (stated honestly)

The reviewer asked for a conservation law. Precision matters here:

- **Evidence is conserved.** E_t is grow-only (Part II, semilattice);
  nothing downstream can create, destroy, or alter an observation. The
  invariant "no decision without an evidence path" (A5 + A6) is a *safety
  property* guaranteed by construction — calling it a conservation law in
  the Noether sense would be dressing; it is an invariant, and that is
  enough.
- **Beliefs are not conserved — deliberately.** Between observations they
  decay toward the prior (A4). The theory's motto: *beliefs decay; evidence
  is permanent.* Forgetting applies to the model of the learner, never to
  the record of the learner.

On "the framework has time but not history": in this formalism the
objection dissolves — **the state is the history** (§1.2, Λ_t is a view of
the timestamped evidence set). A structure learned months ago is decayed in
B_t but intact in E_t; any operator (review policy, stage assessment,
re-fitting of θ) may re-consume it. Explicit H_t-as-extra-argument is
unnecessary when S ≡ history.

---

## 6. The policy, restated as an experiment-design problem

v0 (running): utility = expected mastery gain + novelty + variety, over the
ZPD band, with probe/abstain (Part II §3.4).

v1 (the theory's intended objective): select the next **linguistic
experience** to maximise expected information gain about Λ_t —

```
    a* = argmax_a  E_{o ~ p̂(a)} [ IG( Λ_t ; o, a ) ]
```

subject to the ZPD constraint. Under the Beta upgrade (O2), IG has closed
form per node, making v1 implementable without sampling. The pedagogical
reading: instruction is *optimal experiment design on a learner*, with the
developmental firewall as the ethics constraint (never frustration-zone
experiments). **[objective defined; awaits O2]**

---

## 7. The grand challenge and its falsifiable programme

> **Can second-language development be modelled as the evolution of an
> evidence-grounded linguistic cognitive state — precisely enough that the
> model's parameters are fittable, its predictions calibrated, and its
> instructional decisions justified by expected information gain?**

Hypotheses (extending H1–H4 of Part I, which stand):

- **H5 (tier coupling).** Mastery in V_Φ and V_Χ carries predictive
  information about learning rate in V_Μ beyond within-tier history
  (cross-tier transfer exists and is measurable as improved held-out
  prediction when tier covariates are added). *Testable on existing logs
  once cohort volume suffices.*
- **H6 (L1 signature).** Bengali-L1 and Hindi-L1 learners show distinct,
  CAVP-predicted error distributions on V_Φ contrasts (e.g. /v–w/); the
  contrastive priors beat uniform priors on held-out diagnosis. *This is
  the theory's most linguistically distinctive prediction.*
- **H7 (attrition law).** Post-mastery decay of unpractised nodes follows
  the Δ form with a fittable τ_f; fitted τ_f differs by tier (phonology vs
  lexis). *Directly readable from the ledger.*

**Venues, in order of fit:** BEA workshop (ACL) — the bridge venue for
computational SLA + education; AIED/EDM for H1–H4; *Studies in Second
Language Acquisition* / *Language Learning* for H5–H7 once data exists.
The formalism papers (Parts II–III) stand alone without cohort data; the
empirical papers require the school pilot.

---

## 8. Inclusion & execution ledger

Adopted from the reviews into the theory (this document): the primitive
framing (Λ_t as the object of study), belief-revisions-not-answers, the
linguistic typing of state and evidence, linguistic inference as hypothesis
revision, acquisition-dynamics positioning, EIG policy objective,
grand-challenge statement.

Already existed (review converged on it independently — recorded as
validation, not novelty): state-as-primitive and operators-not-arrows
(Part II §1–§3), traceability invariant (A5/A6), history (state = log).

Rejected / corrected: "conservation law" softened to invariant (§5);
"framework lacks history" (§5); the (P,S,M,R,D,U) tuple pruned to tiers
that actually have content, with zeros published (§1.1).

Execution status:
- `scripts/lambda_state.py` — computes Λ_t projections (per-tier mastery,
  coverage, evidence counts) for any student from the live twin DB.
  **[built this session; research instrument, not yet a product surface]**
- Tier partition derived from existing node/edge types — no schema change.
- Next build items, in dependency order: O2 Beta upgrade (unlocks U_t and
  v1 policy) → V_Σ seed content (unlocks H_t for the tense example) →
  cause-net over V_Φ with CAVP priors (H6 instrument).
