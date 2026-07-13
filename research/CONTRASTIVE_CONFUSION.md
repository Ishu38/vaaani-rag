# Contrastive L1 confusion edges — inhibitory CASCADE links

**Mechanism + validation · 2026-07-13** ·
[`contrastive_confusion.py`](contrastive_confusion.py) ·
results: [`contrastive_confusion_results.json`](contrastive_confusion_results.json)

## The mechanism

Contrastive Analysis (Lado 1957) + the Speech Learning Model (Flege) predict a
learner assimilates an L2 sound their L1 lacks to the nearest L1 category:
Bengali /z/→/dʒ/ ("jero"), Hindi /v/~/w/, both /θ/→/t̪/. We model each as an
**inhibitory CASCADE edge** `target ⟿ attractor` carrying a belief
P(substitution). Unlike an ordinary edge, it is **suppressed as the target
contrast is mastered** — *contrastive percolation*: acquiring /z/ drives the
/z/→/dʒ/ substitution belief down.

`backend/l1_confusion.py` — per-L1 confusion table on the real phoneme nodes
(bn/hi calibrated from Indian-English contrastive phonology; ta/te + a default
set literature-default). Belief starts at the population prior, rises on a
missed production, and is suppressed by correct productions and by mastering the
target node.

## Wired end-to-end (and fired in prod)

- **`ear`** — a per-phone outcome on a confused phoneme moves the belief.
- **`/loop/evidence`** — a phoneme attempt (with `meta.l1`) updates the belief
  *before* diagnosis, so the cause-net reads it.
- **`cause_net`** — `l1_interference` is now driven by the live confusion belief
  (graded, per-phoneme) and the diagnosis **names the substitution** in plain
  language ("the z → dʒ swap that Bengali speakers make…").
- **`/loop/confusion/{sid}?l1=`** — exposes the confusion field.

Public-prod fire (`api.vaaani.in`): baseline /z/ 0.80 → three misses raise it to
**0.92** with `l1_interference` diagnosed and z→dʒ named → five correct suppress
it to **0.02**. Shared /m/ → no substitution.

## Validation — is l1_interference now discriminative?

Simulate 60 learners per L1; drive a miss on each confused and each shared
phoneme; run the real `cause_net.diagnose`; measure how well the l1_interference
posterior separates the two.

| L1 | confused vs shared | mean l1 posterior (confused) | (shared) | AUC |
|---|---|---:|---:|---:|
| Bengali | 5 vs 9 | 0.450 | 0.045 | **1.000** |
| Hindi | 6 vs 10 | 0.450 | 0.045 | **1.000** |

**Reading (honest):** the confusion field turns `l1_interference` into a clean
per-phoneme attribution — a missed /z/ for a Bengali learner is diagnosed as L1
transfer (0.45) while a missed /m/ is not (0.045). The AUC is 1.0 because the
attribution is *structured*, not a noisy prediction: the old binary script-based
heuristic could never fire here (phoneme node displays are ASCII, so script
detection is always false), so this is the signal that makes per-phoneme L1
diagnosis possible at all — not a claim of predictive accuracy on real audio.

## Next

Confusion weights are literature/prior-seeded; the substitution is currently
inferred from a miss on the target (not yet from the *produced* phone, which the
CAVP engine could supply for direct evidence). Calibrate the priors and add
produced-phone evidence once real pronunciation logs accumulate.
