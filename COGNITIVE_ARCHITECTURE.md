# Vaaani — Adaptive Cognitive Architecture for Language Development (v2)

Canonical topology (Neil, 2026-07-12). The system is a hybrid deliberative agent
(AIMA ch. 2) — a closed perception–decision–action–reflection loop around a
continuously evolving learner model. The LLM appears in exactly one box, as a
phrasing device. **Reference frames:** Russell & Norvig, *AIMA* 4th ed. (agent
architecture, probability, decision theory) and Kurdi, *NLP and Computational
Linguistics* vol. 1 (the formal machinery of each linguistic tier: speech ch. 2,
morphology ch. 3, syntax ch. 4).

```
              PERCEPTION
                   │
                   ▼
          Evidence Object Graph
                   │
                   ▼
        Neuro-Symbolic Reasoner
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
 Linguistic World      Cognitive Twin
      Model               Engine
         │                   │
         └─────────┬─────────┘
                   ▼
          Development Engine
                   │
                   ▼
      Pedagogical Planning Engine
                   │
                   ▼
     Procedural Activity Generator
                   │
                   ▼
        Learner Interaction Loop
                   │
                   ▼
         Metacognitive Evaluation
                   │
                   └──────────────┐
                                  ▼
                       Update Cognitive Twin
```

**Status legend:** ✅ shipped & demo-verified · 🧩 exists as module, needs rewiring
· 🔨 to build.

---

## 1 · PERCEPTION 🧩

Transforms raw input into candidate observations. Existing: `extractor.py`,
`ingest.py`, Tesseract OCR (eng/hin/ben), `audio/`, `intent.py`; CAVP's 8-layer
acoustic engine is the mature speech-perception subsystem one product over.
**Kurdi ch. 2** governs the speech tier (phone inventories, grapheme–phoneme
mapping) — the Sound Lab's percepts should emit evidence at the phoneme-node
level, not just word level. **Contract change:** every perceptual module's output
is exactly one thing — EvidenceObjects. No percept ever reaches a prompt directly.

## 2 · Evidence Object Graph ✅

`backend/evidence_graph.py` (shipped 2026-07-12). Typed record:
`(student, node_id, source, outcome, confidence, ts, meta)` — the only currency
between perception and cognition. SQLite store, indexed per (student, node),
linked to Linguistic World Model nodes. AIMA framing: the percept-to-evidence
boundary; each record is one observation for the twin's filtering update (ch. 14).

## 3 · Neuro-Symbolic Reasoner 🧩

Interprets evidence against explicit linguistic rules before it touches the twin
— is this error *phonological, morphological, syntactic*; is it L1 transfer?
Existing seeds: `cognitive/` (classifier, detector, fingerprint), Hermes pre-flight
corrections, the KB-first Clarity-Coach discipline. **Kurdi supplies the symbolic
half per tier:** two-level morphology / FSTs (ch. 3) to analyze a misspelled or
misderived word into its intended morphemes; phonotactic FSAs (ch. 2) to detect
L1-driven substitutions (with CAVP's calibrated Bengali/Hindi catalogs as priors);
CFG/unification parsing (ch. 4) to localize syntax errors. **Build:** the cause
layer — a small Bayes net (AIMA ch. 13) over {L1_transfer, orthographic
interference, developmental stage, slip}, posterior attached to the evidence
before the twin ingests it.

## 4a · Linguistic World Model ✅ (store) / 🔨 (enrichment)

`data/graph.json` — 785 nodes, 1,178 edges — read through
`development_engine.WorldModel`. Edge audit (2026-07-12): 272 `prerequisite_for`
+ 3 `depends_on` edges now drive readiness; `sounds_like` (253), `word_family`
(150), `translates_to` (135), `root_of` (95) are the Kurdi tiers already in data
form — speech, morphology, and cross-language bridges respectively.
**Build:** prerequisite enrichment — Meisel's acquisition orders + the four
strands' internal ordering, so readiness stops defaulting to 1.0 for most nodes.

## 4b · Cognitive Twin Engine ✅

`backend/cognitive_twin.py` (shipped 2026-07-12). Bayesian Knowledge Tracing —
a two-state HMM per (student, graph node) — with: soft evidence weighted by
perceptual confidence, learning step (transit), exponential forgetting toward
the prior (half-life 14 d, so review urgency falls out of the state itself),
and a prediction ledger for metacognition. Demo-verified dynamics: mastery
0.10 → 0.43 → 0.81 → dips to 0.42 on an error → recovers → 0.99 mastered.
This object IS the learner; every other box reads or writes it.

## 5 · Development Engine ✅

`backend/development_engine.py` (shipped 2026-07-12). Makes the ZPD numeric:
`p_success = guess + (1−guess−slip)·(0.35·mastery + 0.65·prereq_readiness)`;
frontier = unmastered nodes with p_success ∈ [0.60, 0.80], best-first,
structure-informed nodes preferred among ties. Below band = frustration zone —
the developmental firewall's territory (`developmental_firewall.py` stays as the
hard constraint). The v0 blend is a declared heuristic; the calibration table
(box 9) is what tunes it against reality.

## 6 · Pedagogical Planning Engine ✅

`backend/pedagogical_planner.py` (shipped 2026-07-12). Decision theory, not an
LLM (AIMA ch. 16): `score = 1.0·expected_gain + 0.6·review_urgency + 0.2·novelty`,
argmax over frontier + decayed-mastery review candidates. Every decision carries
a human-readable reason ("in ZPD (P=0.68), expected gain 0.14, readiness 1.00")
and commits its P(success) to the prediction ledger *before* the learner attempts
— the planner goes on record. Replaces the Discovery-Orchestrator-as-system-prompt.

## 7 · Procedural Activity Generator 🔨

Turns a MissionDecision into learner-facing content. **Kurdi makes this mostly
symbolic, not generative:** morphology drills by FST composition (root × affix
lattice from `root_of`/`word_family` edges, ch. 3); minimal-pair sound activities
from `sounds_like` edges (ch. 2); syntax scrambles validated by the CFG before
they're shown (ch. 4 — never present an unparseable "correct answer"). Template +
world-model facts first; the small LLM phrases child-friendly wording only.
This box is what kills the 4B in the hot path: template ≈ 0 ms.

## 8 · Learner Interaction Loop 🧩

The SPA (word-web UI, Socratic chat, Sound Lab). Contract: every completed
interaction emits EvidenceObjects — the UI is a perception device on its
return path. Existing tracking hooks need rerouting from ad-hoc state to
`evidence_graph.record()`.

## 9 · Metacognitive Evaluation ✅ (v0) / 🧩 (Hermes merge)

`cognitive_twin.calibration()` (shipped): reliability table of the planner's
predicted P(success) vs actual outcomes — the system continuously audits whether
its own confidence is honest (AIMA ch. 3 metareasoning / ch. 28 bounded
rationality). Merge target: Hermes (trace store, k-NN corrections) +
`citation_fidelity()` become the same layer's language-side monitors.
**Abstention rule to add:** flat diagnosis posterior → "show me one more"
instead of guessing.

## 10 · Update Cognitive Twin ✅

The closing arc is `cognitive_twin.update()` — already the loop's re-entry point.
Demo (`backend/demo_cognitive_loop.py`) runs the full cycle on the real graph:
evidence → twin → frontier → mission (+logged prediction) → attempts →
calibration. Run: `cd backend && python demo_cognitive_loop.py`.

---

## Edge AI consequence

Boxes 2, 4b, 5, 6, 9 — the entire decision core — are stdlib arithmetic over
SQLite/JSON (shipped today; no torch, no network). Box 7 is templates + FSTs.
The only heavy component left is phrasing (box 7's last step) and free chat:
a 1.5–2B int4 on-device model (WebLLM/WebGPU; transformers.js fallback), cloud
only as rare escalation. Learner state → IndexedDB. The GCP 4B exits the hot
path entirely — before the free tier ends Oct 3.

## Build order from here

1. **Wire real flows into evidence** (box 1/8): quiz + review + mission handlers
   call `evidence_graph.record()` / `cognitive_twin.update()`. First real learner
   data starts accumulating immediately.
2. **Activity Generator v0** (box 7): templates for the three Kurdi tiers over
   existing edge types; 4B leaves the mission path.
3. **Prerequisite enrichment** (box 4a): Meisel-seeded edges; frontier becomes
   discriminating instead of permissive.
4. **Cause Bayes net** (box 3): the diagnosis layer, CAVP L1 priors.
5. **Hermes merge + abstention** (box 9).
6. **Edge packaging** (WebLLM + IndexedDB).

Rule per phase: wired end-to-end — evidence visible in the twin, decision visible
in the SPA — before the next phase starts.

## Research claims this architecture can make (and an LLM wrapper cannot)

- Calibration curves of its own learner model (box 9 table — falsifiable).
- Cause-attribution accuracy vs expert linguist diagnosis (box 3).
- Learning-gain delta: decision-theoretic vs random activity selection (box 6,
  A/B-able per cohort).
- Full pedagogy runs on-device with no model above 2B — the architecture, not
  the parameter count, is the contribution.
