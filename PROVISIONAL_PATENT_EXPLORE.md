# Provisional Patent — Disclosure Draft
## "System and Method for Camera-Initiated, Developmentally-Gated Language Acquisition with Incremental Per-Learner Concept-Graph Construction"

> **What this is:** a technical disclosure draft for your patent attorney (the one
> who filed the MLAF gesture-grammar application). It is written to be the *cheap*
> half of the job — the inventor's complete technical description — so the attorney
> only has to do claim-drafting and §3(k) framing, not discovery. It is **not legal
> advice.** Confirm patentability strategy, claim scope, and the §3(k) technical-
> effect argument with the attorney before filing.
>
> **Why this is the patent-eligible piece (and the algorithm is not):** this claims
> an *apparatus + method coupled to a camera and on-device inference producing a
> technical effect* — the same reason MLAF is defensible. It is **reduced to
> practice** (a working system exists). The unbuilt developmental-learner model
> (Path 2) is deliberately **excluded and kept as a trade secret** — patenting it
> would only teach it, and it isn't reduced to practice.

---

## 1. Field of the Invention

Human-computer interaction and educational technology; specifically, a camera-
initiated interactive system and method for language acquisition in which a
real-world object captured by a device camera drives a developmental-stage-gated
adaptive dialogue and incrementally builds a persistent, privacy-scoped, per-
learner concept graph.

## 2. Background & Technical Problem

Existing language-learning software presents pre-authored content on a screen; the
learner is a *consumer* of curriculum decoupled from their physical environment.
Where cameras are used (e.g. augmented-reality translators, object-labelling
apps), the interaction terminates at naming the object — a single lookup — and
does not (a) elicit staged language *production* from the learner, (b) adapt the
interaction to the learner's developmental stage with a technical output
constraint, or (c) accumulate the interaction into a persistent, individualized
knowledge structure.

Three unsolved technical problems follow:

- **P1 — Privacy vs. recognition.** Object recognition capable of running on a
  child's live camera has conventionally required transmitting image frames to a
  server, creating a data-exposure problem, particularly for minors.
- **P2 — Developmental appropriateness of generative output.** A generative
  language model produces output unsuitable for young learners (e.g. phonetic
  notation, abstract symbols). No deterministic, real-time, streaming-safe
  mechanism exists to constrain such output as a function of the learner's
  developmental stage without discarding the generative benefit.
- **P3 — Ephemeral interaction.** Camera-and-teach interactions are ephemeral;
  they do not construct a durable, per-learner representation of what has been
  acquired and how strongly it is retained.

## 3. Summary of the Invention

A system and method wherein:

1. A **device camera** captures a frame of a real-world object.
2. An **on-device recognition model** (executing locally, such that image frames
   are not transmitted off the device) produces a set of ranked candidate object
   labels — solving **P1**.
3. A **confirmation step** presents the ranked candidates and receives a learner
   selection, so a recognition error is never asserted to the learner as fact
   (a technical error-containment safeguard converting a mis-recognition into a
   selectable prompt).
4. The confirmed object seeds an **ordered, stage-gated adaptive dialogue** whose
   stages correspond to an acquisition progression — Recognize → Construct →
   Reason → Communicate — each stage gating advancement to the next and
   incrementing a per-concept mastery value.
5. Generative dialogue output is passed through a **developmental-stage output
   firewall**: a deterministic, streaming-boundary-safe transformer that, as a
   function of a learner developmental-stage signal, suppresses or rewrites
   designated symbol classes (e.g. phonetic notation) in real time across the
   token boundaries of a model output stream — solving **P2**.
6. The **Communicate** stage elicits a learner-produced media artifact (a
   recorded narration), stored under a per-learner privacy scope; completion
   consolidates the concept to full mastery — the system elicits language
   *production*, not consumption.
7. Each completed discovery **incrementally constructs a persistent, privacy-
   scoped per-learner concept graph**: the object becomes a node whose rendered
   visual state (size/brightness) is derived from an accumulated mastery value
   and from a spaced-repetition memory-strength model, merged into a single
   structure alongside curated-curriculum nodes — solving **P3**.

## 4. Detailed Description (with Figure references)

**FIG. 1 — System architecture.** A learner device (110) comprising a camera
(111), an on-device recognition model (112), a display (113) and a microphone/
recorder (114); a server (120) comprising a dialogue-state engine (121), a
developmental output firewall (122), a per-learner concept-graph store (123), a
privacy-scope resolver (124), and a spaced-repetition memory model (125).

**FIG. 2 — Method flow.** Capture (201) → on-device recognition producing ranked
candidates (202) → confirmation selection (203) → discovery record created under
learner scope (204) → stage-gated dialogue loop (205) → developmental firewall on
each generated turn (206) → Communicate media capture (207) → concept-graph node
construction / mastery update (208).

**FIG. 3 — Developmental output firewall.** A streaming scrubber (300) maintaining
a hold-buffer across token boundaries; on receipt of an opening delimiter of a
protected symbol class it withholds emission until the class can be
disambiguated, substituting a plain-language equivalent when confirmed, as a
function of a developmental-stage threshold (301) parsed from the learner context.

**FIG. 4 — Per-learner concept graph.** Curated-curriculum nodes (410) and
camera-discovered nodes (420) merged into one privacy-scoped structure; a self-
discovered cluster ("My World", 430); node visual state (size/brightness) computed
from mastery and memory-strength (440); nodes due for reinforcement rendered with
a distinct marker (450).

**Embodiments and variations** (to broaden coverage): the recognition model may be
an object detector or image classifier; the developmental-stage signal may be an
explicit grade, an inferred age, or a mastery-derived level; the protected symbol
class may be phonetic notation, orthographic complexity, or lexical difficulty;
the produced artifact may be video, audio, or a drawing; the concept graph may be
per-learner, per-cohort, or per-classroom under the privacy-scope resolver; the
stage progression may be reordered or extended while preserving the gate-and-
increment property; the recognition may run wholly on-device or in a hybrid where
only derived, non-image features leave the device.

## 5. Draft Claims (for the attorney to refine)

*Independent — method:*

> **1.** A computer-implemented method for language acquisition, comprising:
> capturing, by a camera of a device, an image of a physical object; producing,
> by a recognition model executing on the device such that the image is not
> transmitted from the device, a plurality of ranked candidate labels for the
> object; receiving a learner selection of a label; initiating an ordered
> multi-stage dialogue seeded by the selected label, wherein each stage must be
> completed to advance and each completion increments a per-concept mastery
> value; transforming generated dialogue output, in real time and across output-
> stream token boundaries, as a function of a learner developmental-stage signal,
> to suppress or rewrite a designated symbol class; eliciting, at a terminal
> stage, a learner-produced media artifact; and constructing, from the completed
> dialogue, a node in a persistent per-learner concept graph whose rendered
> visual state is derived from the mastery value and a memory-strength model.

*Independent — apparatus:*

> **2.** A system comprising a device having a camera and an on-device recognition
> model, and a server having a dialogue-state engine, a developmental output
> firewall, a privacy-scope resolver, and a per-learner concept-graph store,
> configured to perform the method of claim 1.

*Dependent (the defensible novelty lives here — narrow, combination-specific):*

> **3.** …wherein the recognition model runs wholly on-device and image frames are
> never transmitted, the ranked candidates being the sole output leaving the
> recognition step. *(technical effect: privacy + bandwidth — P1)*
>
> **4.** …wherein the transforming comprises a streaming scrubber maintaining a
> hold-buffer that withholds emission upon an opening delimiter of the designated
> symbol class until the class is disambiguated across token boundaries.
> *(technical mechanism — P2)*
>
> **5.** …wherein the designated symbol class is phonetic notation and the
> developmental-stage signal defines a threshold grade below which said notation
> is suppressed, said threshold being bounded within a fixed range. *(the
> grade-gated notation firewall)*
>
> **6.** …wherein the multi-stage dialogue stages correspond to an ordered
> acquisition progression of recognition, construction, reasoning, and
> communication, the communication stage eliciting the learner-produced artifact.
>
> **7.** …wherein the concept graph merges curated-curriculum nodes and camera-
> discovered nodes into a single privacy-scoped structure, camera-discovered nodes
> forming a distinct learner-owned cluster.
>
> **8.** …wherein a node's rendered size or brightness increases with the mastery
> value and a node is marked for reinforcement when a spaced-repetition interval
> for the concept has elapsed.
>
> **9.** …wherein the confirmation step converts a recognition below a confidence
> threshold into a selection prompt rather than an assertion. *(error-containment
> safeguard)*

## 6. Statement of Technical Effect (the §3(k) argument)

The invention is not a computer programme *per se*: it is a camera-coupled
apparatus and method producing concrete technical effects — (i) on-device
recognition that keeps image data on the device (a data-security and bandwidth
effect, P1); (ii) a deterministic streaming-boundary-safe mechanism controlling a
model output stream in real time as a function of a state variable (a human-
machine-interaction control effect, P2); (iii) a specific method of incrementally
constructing and privacy-scoping a per-learner data structure whose rendered state
is computed from mastery and a memory model (P3); and (iv) an error-containment
safeguard in a recognition system (claim 9). Per *Ferid Allani v. Union of India*
and the CRI guidelines, software demonstrating such technical contribution falls
outside the §3(k) bar.

## 7. Industrial Applicability

Educational technology for language and literacy, particularly for multilingual
and young learners; deployable on consumer mobile devices without specialised
hardware.

## 8. Abstract

A camera on a learner's device captures a real-world object; an on-device model
recognises it without transmitting the image; the learner confirms from ranked
candidates; the confirmed object seeds a stage-gated adaptive dialogue advancing
through recognition, construction, reasoning and communication, each stage
increasing a per-concept mastery; generated output is constrained in real time by
a developmental-stage firewall that suppresses designated symbols across output-
stream boundaries; a terminal stage elicits a learner-produced narration; and each
discovery incrementally builds a persistent, privacy-scoped per-learner concept
graph whose visual state encodes mastery and memory strength.

---

## Filing notes (strategic — read before you file)

- **Take this to the MLAF attorney.** It is the same claim shape (apparatus +
  method, hardware-coupled, technical effect). You already ran this playbook.
- **Keep the claims narrow** — around the *combination* in claims 3–9, not the
  broad idea of "camera + teach," which has prior art (AR translators, Lens-style
  labelling apps). The novelty you can defend is the union of: on-device-only
  recognition + confirmation safeguard + gate-and-increment acquisition stages +
  the streaming developmental firewall + the merged privacy-scoped mastery graph.
- **Do a prior-art sanity check** with the attorney against AR language apps and
  vocabulary-camera apps before spending on the non-provisional.
- **Exclude the Path-2 developmental model.** It is unbuilt and its value is
  secrecy. Do not disclose it here or anywhere public.
- **The provisional starts a 12-month clock** to the complete specification. Only
  file when you're prepared to fund the non-provisional within a year — and file
  it as *insurance alongside shipping*, not instead of it. A patent protects an
  asset; only shipping creates one.
- **Also cheap and overlooked:** a **registered design** (Designs Act) on the
  "My Universe" visual interface, and **character/trademark** protection on the
  Core Library characters (Tri, Uni, …). Different rights, dodge §3(k) entirely.
