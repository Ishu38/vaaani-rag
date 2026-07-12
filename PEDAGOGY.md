# Vaaani — The Linguistics Tutor for Grades 1–8
## Pedagogy & Curriculum Charter (the Root Bridge Method)

*This document is the canonical answer to "what is Vaaani's pedagogy?" Every
claim in it is either implemented in the product today or explicitly marked
ROADMAP. Speak from it freely.*

---

## 1. What Vaaani is

Vaaani is a linguistics tutor for school students (Grades 1–8) built for
India's multilingual classrooms. It does not teach "English grammar rules to
memorise." It teaches children **how language itself works** — sounds, word
structure, sentences, meaning, and the histories that connect English to the
languages spoken at home — and it adapts that teaching to each individual
child through diagnosis, not through a fixed sequence.

The one-sentence version:

> **"Vaaani teaches children how language works — using the language they
> already speak at home as the bridge."**

Built by an applied linguist (MA Applied Linguistics, 14 years of teaching;
AI/ML certification, IIT Patna). The pedagogy is not a wrapper around a
chatbot; it is enforced in the engine's code, which is the honest answer to
"how is this different from ChatGPT?"

---

## 2. The Root Bridge Method — four principles

Each principle states the idea, the learning science behind it in plain
terms, and the mechanism that enforces it in the product. The enforcement is
the exclusive part: most edtech *promises* pedagogy; Vaaani's is *compiled in*.

### Principle 1 — Sound before Symbol

**The idea.** Children build phonemic awareness by ear years before they can
handle abstract notation. Awareness of sounds — clapping syllables, hearing
that 'thick' has three sounds while having five letters — is among the
strongest known predictors of reading success. Notation (IPA symbols,
slash-brackets) is a Grade 5+ tool; used earlier it turns play into decoding.

**The enforcement.** The engine carries a deterministic notation firewall at
the serving layer: below the school's notation gate, any phonetic notation
the model attempts to emit is rewritten into plain-language sound talk before
it reaches the child — even mid-stream. This is not a prompt instruction the
AI might forget; it is code that cannot be talked out of it.

**The gate is a bounded dial, not a fixed number.** Default: Grade 5.
Schools may set it anywhere in **Grades 4–6** (ICSE-style early-notation
boards → 4; longer sound-play boards → 6). The floor of 4 is deliberate and
NOT configurable: phonemic awareness is auditory, and below that age
notation converts listening into decoding.

**Say it as:** *"Below your school's notation gate — Grade 5 by default —
Vaaani physically cannot show your child a phonetic symbol. Your board can
move the gate between Grades 4 and 6. Below 4, the engine won't budge — and
that refusal is exactly why you can trust the rest of it."*

### Principle 2 — Discovery before Definition

**The idea.** Children retain patterns they find themselves far better than
definitions they are handed (guided discovery; Bruner's spiral; the
generation effect). A linguistics tutor should behave like a field
scientist's mentor: put data in front of the child, ask the question that
makes the pattern visible, and only then let the child name it.

**The enforcement.**
- **Socratic mode** is a first-class engine mode, not a persona prompt. In
  Socratic mode the tutor is instructed — per linguistic level — to withhold
  the answer: for sounds, "say it aloud, what is your mouth doing?"; for
  words, "peel the word apart, what does each piece contribute?"; for
  sentences, "move the chunk, what breaks?"; for meaning, "could this
  sentence mean two things?"
- **Schools can make Socratic the enforced default** for their students
  (guardrails), and every answer is automatically checked for violations of
  the school's direct-answer policy.
- In the Root Bridge word units, **technical jargon (root, Latin,
  etymology…) is forbidden through Grade 4** — the child meets a character,
  hears a story, spots the shared piece across a word family, and *discovers*
  the secret. The word "root" arrives only after years of finding roots.

**Say it as:** *"Vaaani doesn't tell your child that 'tri' means three. It
introduces Tri, who turns everything into three — and lets the child catch
'triangle, tricycle, trio' red-handed."*

### Principle 3 — The Mother Tongue is a Bridge, not a Barrier

**The idea.** Indian children arrive with one to three languages already in
their heads. Traditional English teaching treats this as interference to be
suppressed. Fifty years of research on additive multilingualism — and NEP
2020's mother-tongue emphasis — say the opposite: the home language is the
single richest resource for learning how language works.

**The enforcement.**
- **Every Root Bridge unit carries a home-language connection** (Hindi and
  Bengali today), and the curriculum data distinguishes — with linguistic
  honesty — between a **cognate** (English 'tri-' and Sanskrit 'त्रि' really
  are the same ancient word; safe to tell a child "these are cousins") and a
  mere **translation** (never claimed as shared ancestry). Most products
  would fudge this; ours cannot, because the distinction is in the data
  schema.
- The **Cognitive X-Ray** (see Principle 4) has *mother-tongue transfer* as a
  first-class diagnosis. When a Bengali-medium child writes 'van' for 'wan',
  Vaaani's remediation says, in effect: *"this is your mother tongue's rule
  showing through — a normal stage, not a mistake in thinking. Here are the
  two rules side by side."* The child learns contrastive awareness instead
  of shame.
- The tutor's answers must be **plain-language enough for a parent who
  doesn't read English fluently** to follow the gist — a standing product
  rule.

**Say it as:** *"When your child says 'iskool', that's not broken English —
that's Bangla's sound rules doing exactly what they should. Vaaani is the
first tutor that tells the child that, and then teaches the English rule
next to it."*

### Principle 4 — Evidence before Advancement

**The idea.** "Individual learning development" is an empty phrase unless
the system can say *what* is weak, *why*, and *what changed*. Two children
who miss the same question can need opposite help: one memorised a
definition without the concept; one knows the rule and over-applied it.
Advancement should follow demonstrated mastery, spaced over time — not
lesson completion.

**The enforcement — the individualisation loop:**
1. **Diagnose.** Every practice answer runs through the Cognitive X-Ray,
   which classifies the *thinking pattern*, not just right/wrong, using a
   linguistics-specific error taxonomy: conceptual gap, rote memorisation
   override, term mix-ups (phoneme vs morpheme), **letters-vs-sounds
   conflation**, **mother-tongue transfer**, **rule overgeneralisation**
   (the child who says 'goed' is praised for rule-learning and drilled on
   the rebels), impulsivity, overconfidence/underconfidence, fragile
   understanding.
2. **Adapt.** The student's weak spots are injected into the tutor's
   Socratic questioning ("bias your questions toward these"); the practice
   engine adapts difficulty question-by-question and coaches test-taking
   behaviour (impulsive streaks, tunnelling, confidence calibration) in
   real time.
3. **Consolidate.** A graph-aware spaced-review engine resurfaces each
   concept exactly when it is about to fade, as cloze passages cut from the
   child's own study material; decks export to Anki for offline drill.
4. **Evidence.** Per-topic mastery, error-pattern fingerprints, and
   confidence calibration accumulate into a profile the tutor remembers
   across sessions — and that teachers and parents can see (below).

**Say it as:** *"Vaaani doesn't mark answers wrong. It figures out which of
twelve kinds of wrong it was — and the fix is different for each."*

---

## 3. The curriculum — four strands, spiralled across four bands

Linguistics has natural levels; the curriculum walks all four **as strands in
parallel**, revisiting each at greater depth every band (a spiral, not a
staircase). Nothing is "covered once"; everything returns.

**The four strands:**
- **A. Sounds** (phonetics & phonology)
- **B. Words** (morphology & etymology — the Root Bridge core)
- **C. Sentences** (syntax & grammar-as-pattern)
- **D. Meaning & Use** (semantics & pragmatics)

### Band 1 · Grades 1–2 — *Language as play*
- **Sounds:** syllable clapping; rhyme families; first-sound games; "does your
  voice buzz?" (voicing by touch, no terminology).
- **Words:** Root Bridge Grade 1–2 roots (numbers, body, nature characters:
  Tri, Uni, Bi, Octo, Cent, Denta…): character → story → picture → spot the
  family → grow one new member → the secret + the home-language cousin.
  Plurals as a game ("one cat, two cat__?").
- **Sentences:** who-did-what with picture sentences; silly-order sentences
  ("ate the mango I") the child repairs by ear.
- **Meaning:** opposites, "words that mean almost the same", naming feelings
  a word carries.
- **Hard rules:** no notation, no jargon, story-led, celebrate everything.

### Band 2 · Grades 3–4 — *Language as pattern*
- **Sounds:** minimal pairs by ear (ship/sheep); counting sounds vs letters;
  stress by exaggeration (ba-NA-na); still **zero notation** (firewall).
- **Words:** Root Bridge Grade 3 roots + Grade 4 *combine/compare/transfer*
  (no new roots in G4 — the spiral's consolidation year; old roots recombine:
  known piece + known piece = decodable new word). Prefixes/suffixes as
  "word machines"; fun exceptions (goed/went) as detective cases.
- **Sentences:** subject/verb/object as jobs, not labels; question-making;
  the moveable-chunk test; English order vs Hindi/Bengali order discovered
  by translating one sentence both ways.
- **Meaning:** idioms as treasure ("kick the bucket" — why can't you work it
  out from the words?); polite asking ("could you pass the salt" is not a
  question about ability).

### Band 3 · Grades 5–6 — *Language as system* (notation unlocks)
- **Sounds:** IPA introduced as a tool the child has earned; slashes vs
  brackets; place & manner via mirror work; why 'iskool' happens
  (phonotactics + prothesis — the child's own accent becomes data).
- **Words:** free vs bound morphemes; derivation vs inflection; the famous
  eight inflectional endings; word-formation zoo (blends, conversion,
  borrowings); etymology proper — jungle, shampoo, bungalow, curry, and what
  "borrowed" vs "inherited" means.
- **Sentences:** phrases as trees (drawn!); relative clauses; passive voice
  as perspective shift; SOV vs SVO across the languages the class speaks.
- **Meaning:** synonym/antonym/hyponym as a mapped network; homonymy vs
  polysemy with the connectedness test; ambiguity hunting ("visiting
  relatives can be boring").

### Band 4 · Grades 7–8 — *Language as science*
- **Sounds:** phonological processes (assimilation: handbag→hambag);
  aspiration and why /p/ in 'pin' ≠ /p/ in 'spin'; syllable structure
  (onset/rime); connected speech.
- **Words:** productive morphological analysis of unseen words; suppletion
  (go/went, good/better); semantic change (nice, awful — amelioration and
  pejoration).
- **Sentences:** structural ambiguity and garden paths ("the old man the
  boats"); case (why "him went" fails); cross-linguistic typology-lite with
  Indian languages as primary data.
- **Meaning & Use:** presupposition ("John stopped smoking"); scope ("every
  student read a book"); indirect speech acts; register — how you speak to
  your friend vs your headmaster, and why both are grammatical; language
  families — Grimm's law walked from pitā/pater/father, Indo-Aryan and
  Dravidian as families, the child's own languages placed on the tree.

**Assessment across all bands** is continuous and low-stakes: the practice
engine's six subject pools (Phonetics, Phonology, Morphology, Syntax,
Semantics, Etymology) are difficulty-rated 1–5 and graded leniently on
meaning, not exact wording; the Feynman explain-it-back mode has the child
*teach the concept back* and diffs their explanation against the concept
map; exam-pressure simulation (Grades 6+) trains calibration and composure,
not just recall.

---

## 4. Individual development — what "each student" concretely means

- **A persistent learner profile**: the tutor remembers the student's
  mastery per topic, error fingerprint, and confidence calibration across
  sessions; Socratic questioning is biased toward their weak spots.
- **Their own corpus**: students and teachers upload the actual textbook,
  notes, even a photo of the blackboard (OCR) — answers are grounded in and
  cited from *that* material, and the tutor honestly refuses questions
  outside it rather than inventing. Review cards are cut from the child's
  own pages.
- **Their own languages**: Hindi and Bengali connections shipped; the
  method extends to any Indic language by adding connection data, not by
  re-engineering.
- **Their own pace**: spiral means a child weak in Sounds and strong in
  Words works at different depths per strand simultaneously — the review
  scheduler and difficulty adaptation handle each strand independently.
- **Privacy by design**: every child's documents, memory, and profile are
  scoped to them and their school; parental consent (DPDP §9) is built in
  for under-18 signups, with parent dashboards for visibility.

## 5. Roles around the child

- **Teacher/School:** school workspace with roles (admin/teacher/student/
  parent), invite codes, shared school corpus (a teacher's uploads are
  visible to their school), guardrail policy (e.g., enforce Socratic mode,
  restrict to curriculum scope), school dashboard, per-student cognitive
  fingerprints.
- **Parent:** consent controls, visibility of progress, and — deliberately —
  tutor answers written in plain language a non-English-fluent parent can
  follow.
- **The tutor itself** runs on Vaaani's own language engine (own weights,
  own fine-tuned curriculum adapter, self-hosted) — a child's data does not
  train or transit a third-party AI provider.

---

## 6. Talking tracks

**To a principal (30 seconds).**
"Every school teaches English; nobody teaches children how language works.
Vaaani is a linguistics tutor for Grades 1–8 that turns your students'
multilingualism into the asset NEP 2020 says it is. It diagnoses *why* a
child got something wrong — mother-tongue transfer, letters-vs-sounds
confusion, over-applied rules — and adapts to each child. Your teachers set
the guardrails; you see the dashboards; the syllabus materials your school
uploads are what it teaches from."

**To a parent (30 seconds).**
"Your child already speaks two or three languages — that's a superpower, and
Vaaani is the first tutor that uses it. It teaches how words, sounds and
sentences work through stories and discovery, connects every English word
back to Hindi or Bangla, remembers exactly what your child finds hard, and
practises it at the right moment. And below Grade 5 it's physically
incapable of showing your child confusing symbols — sound play first."

**To an investor / edtech person (30 seconds).**
"Linguistics-for-schools is an empty category — everyone does exam prep or
spoken English. We hold the pedagogy in code: a serving-layer
age-appropriateness firewall, an enforced Socratic mode, a
linguistics-specific error-diagnosis taxonomy, and a 45-root discovery
curriculum with honest Indic cognate mapping. Own fine-tuned model, own
infrastructure, per-school multi-tenancy with DPDP compliance built in.
The moat isn't the LLM — it's the curriculum data and the enforcement layer
around it."

**Objection handling.**
- *"How is this different from ChatGPT?"* — ChatGPT will hand a Grade 2
  child IPA symbols and a definition of 'morpheme'. Vaaani cannot: grade
  gating, Socratic enforcement, grounding in the school's own materials,
  and error diagnosis are engine features, not prompts. Also: a child's
  chat with ChatGPT trains nothing about the child; Vaaani builds a
  longitudinal learning profile the teacher can act on.
- *"Will it replace teachers?"* — No, and it refuses to try: in school mode
  it asks guiding questions instead of giving answers, and its dashboards
  exist to give teachers diagnostic vision they've never had.
- *"My child studies in Bengali medium."* — Perfect; that's who this is
  built for. The method treats Bangla as the bridge, and the tutor's
  explanations are written to be followable by the whole family.
- *"Screen time?"* — Sessions are short by design (one question, one
  discovery, one review card); audio mode lets a child listen instead of
  scroll; Anki export moves drill offline.

---

## 7. Honest boundaries (never claim past these)

- The **Root Bridge tutor experience** (character/story/discovery flow) is
  trained into Vaaani's flagship model adapter; its full interactive surface
  in the product is **ROADMAP** — the shipped product today is the tutor
  chat (Socratic + grounded), practice simulation, cognitive diagnosis,
  spaced review, audio, Feynman mode, school/parent layer.
- **Efficacy**: the *methods* (phonemic awareness → literacy; retrieval
  practice; guided discovery; additive multilingualism) are established
  learning science; **Vaaani's own outcome studies haven't been run yet**.
  Say "built on", never "proven to".
- Hindi and Bengali connections are shipped; other languages are extensible
  but not yet built.
- No IELTS/exam-score claims — different product, different engine.
- No named-institution endorsements. Credentials that are safe to state:
  MA Applied Linguistics, 14 years' teaching, AI/ML certification (IIT
  Patna).
