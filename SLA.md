# Service-Level Agreement — Grounded Answer Rate

This document is the operational spec for the quality guarantee
referenced in [COMMERCIAL.md](COMMERCIAL.md). It exists in source
control so prospects can verify the claim is measurable and the
mechanism is real, not marketing copy.

> **The guarantee, in one line:**
> ≥ 85 % grounded-answer rate on your corpus, audited weekly. If we
> drop below 85 % for two consecutive weekly audits, you receive a
> pro-rated refund for the affected month, no questions asked.

## What "grounded" means

For every answer the system produces on a knowledge-intent query,
`backend/llm.py :: citation_fidelity()` performs this check:

1. Tokenise the answer into sentences (regex split on `[.!?]`
   followed by whitespace).
2. For each sentence, extract content tokens — alphanumeric words of
   3+ characters, lowercased, after stop-word removal.
3. Tokenise every retrieved chunk the same way and union them into
   one set of `grounded_tokens`.
4. A sentence passes if **≥ 20 %** of its content tokens appear in
   `grounded_tokens`. A sentence fails otherwise.
5. An answer is **grounded** if zero sentences fail. Otherwise the
   answer is **partially-grounded** and a `fidelity_warning` per failed
   sentence is attached to the chat response (visible in the SPA as the
   amber `⚠ N claims aren't grounded in the retrieved sources` band).

The 20 % token-overlap threshold is intentionally permissive — it
catches answers that drift into model-internal-knowledge territory
without falsely flagging legitimate paraphrase. Customers who want a
stricter threshold (e.g. 35 % for legal / medical corpora) can
configure it; the SLA percentage adjusts to compensate.

## How the audit works

A nightly cron on the customer's deployment runs
[`backend/eval.py`](backend/eval.py) against a customer-supplied
`eval_cases.json` file. Each case is one of:

```json
{
  "query": "natural-language question a user might actually ask",
  "expected_intent": "knowledge",                  // optional
  "expected_terms": ["phonogram", "Rule 17"],      // optional
  "expected_source_substring": "logic-of-english"  // optional
}
```

The harness:

- POSTs each query to `/chat` with a long-lived session cookie owned
  by an audit-only service account on the customer's instance;
- records per-case `grounded` (1 if no fidelity warnings, else 0),
  `source_hit`, `mentions_all`, `intent_match`, `latency_ms`;
- writes per-case CSV + aggregate `summary.json` to a path the
  customer mounts (typically `runs/YYYY-MM-DD.csv`).

The customer receives a copy of every nightly run by email or shared
drive; we don't get to grade ourselves in private.

## What the "weekly grounded-answer rate" is

For each calendar week (UTC Mon 00:00 → Sun 23:59) we compute:

```
weekly_grounded_pct = mean( daily.grounded_pct for daily in week )
```

where each `daily.grounded_pct` comes from the harness's
`_summarise()` function — i.e. it's the same number written into that
day's `summary.json`. The four-line implementation is in
[`backend/eval.py`](backend/eval.py); you can audit it in 30 seconds.

## Eval-case quality (the only place you have to trust us)

The eval cases are supplied by the **customer**, not us. Cases must:

- Be drawn from genuinely-asked or genuinely-expected user questions;
- Span at least 5 different source documents (or 80 % of the corpus
  if smaller);
- Include 10 % "should-not-be-answered" cases (out-of-corpus
  questions, deliberate hallucination probes) — these test that the
  system correctly surfaces fidelity warnings instead of fabricating.

If we suspect cases are gamed (e.g. all 25 are the same shape with
unique trivial answers) we will ask for a revision before the SLA
window starts. The customer always has the final say on the case set.

## What triggers the refund

Plain English:

- Each weekly audit produces a single number, `weekly_grounded_pct`.
- If two consecutive weekly numbers are below **85.0**, the SLA
  trigger fires.
- The refund is calculated as: `(subscription_monthly / 30) * days_in_affected_window`,
  where the affected window is from the START of the first
  below-threshold week through the END of the second below-threshold
  week (typically 14 days).
- Refund is processed within 7 working days of the second
  below-threshold audit landing.
- The trigger resets the moment a weekly audit returns ≥ 85.0; one
  bad week sandwiched between two good weeks does **not** trigger.

The refund is automatic. The customer does not need to email,
escalate, or argue.

## What is excluded from the SLA

- Queries whose `intent` is not `knowledge` (calendar / task / meta
  responses are not grounded against the corpus by design).
- Days on which the customer's DeepSeek API key is rate-limited or
  out of budget (we cannot ground an answer the LLM never returned).
- Audits skipped because the customer's deployment was down for
  maintenance (the harness emits `network/HTTP failures` separately
  from grounded%; those don't count against the SLA).
- The first 14 days after a major corpus change (e.g. ingesting >25 %
  new documents) — the entity graph needs time to stabilise.

## Code you can audit right now

| Claim | File you can verify it in |
|---|---|
| Grounding is checked per sentence on every chat response | `backend/llm.py :: citation_fidelity` |
| Warnings are attached to the chat response body | `backend/main.py` (search `fidelity_warnings`) |
| Warnings are shown in the SPA (not hidden) | `frontend/index.html` (search `fidelity-warn`) |
| The eval harness exists and is runnable today | `backend/eval.py` + `backend/eval_cases.json` |
| The harness scoring is the same math as `_summarise()` | `backend/eval.py :: _summarise` |

All five files are in this repository. Read them.

## Why we can offer this and competitors can't

Most RAG products at the ₹6-40 K / month band rely on a black-box
LLM call + a vector store and **hope** the model paraphrases the
retrieved chunks faithfully. They have no in-product mechanism to
detect when it doesn't. Vaaani RAG has had `citation_fidelity` since
the first commit and now surfaces the warnings in the UI; the eval
harness simply systematises what was already running per-response.

The SLA is the contractual surface of an architectural choice that
predates the SLA itself.

---

*Last updated alongside the code that implements it. The single source
of truth for the audit mechanism is the code in this repository, not
this document. Where this document and the code diverge, the code
wins.*
