# Commercial Licensing — Vaaani Graph-RAG

## Who needs to pay

The Business Source License 1.1 (see `LICENSE`) lets you use Vaaani RAG
**free of charge** if you are:

- an individual learner, student, family, or independent researcher;
- a registered non-profit serving education, accessibility, or literacy;
- an accredited school or college using Vaaani in classroom instruction
  with no separate fee charged to the student or guardian;
- a university or hospital using Vaaani for non-commercial academic
  research (citation of the project is requested in any publication).

If you fall **outside** all of these — for example you are a private
clinic that bills patients, a tutoring chain that monetises access, a
content-publisher SaaS, an EdTech platform integrating Vaaani into a
paid product, or any for-profit enterprise — you need a commercial
licence. This file explains the tiers and the quality SLA that ships
with them.

## Pricing (India, FY 2026-27)

Indicative ranges. Final quote depends on user count, support level,
and customisation scope. Quoted in INR; international pricing on
request.

| Buyer profile | Structure | Indicative range |
|---|---|---|
| Single tutor / coaching centre / SLP practice | Annual per-seat | ₹6,000 – ₹18,000 / seat / year |
| Single school (≤ 200 students) | Annual institutional | ₹40,000 – ₹1,20,000 / year |
| Multi-school chain or NGO network | Tiered by site count | ₹3 L – ₹15 L / year |
| Hospital department / clinic group | Annual departmental | ₹1.5 L – ₹6 L / year |
| EdTech / SaaS integration partner | Annual platform + per-active-user | ₹6 L – ₹40 L / year + ₹50–₹200 PAU |
| Hardware OEM bundle | Royalty per device sold | ₹400 – ₹1,500 / device |
| Source-code modification + redistribution | Annual OEM licence | ₹10 L – ₹35 L / year |

## What a commercial licence buys you

| Item | Free tier (BSL) | Commercial tier |
|---|:---:|:---:|
| Source-code access | ✓ | ✓ |
| Self-host the web app | ✓ | ✓ |
| Modify the code | ✓ | ✓ |
| Use in a fee-charging service or product | — | ✓ |
| Multi-user cohort dashboards | — | ✓ |
| Branded learner/student reports (PDF export) | — | ✓ |
| Priority email support (≤ 24h business-day response) | — | ✓ |
| **Grounded-answer-rate SLA (see SLA.md)** | — | **✓** |
| Custom motor / linguistic-profile calibration | — | optional add-on |
| Integration help (Teachmint / Fedena / hospital EMR) | — | optional add-on |
| OEM redistribution rights | — | optional add-on |
| Indemnity against patent claims | — | ✓ |

## The quality SLA — short version

Every commercial customer's deployment runs a nightly eval harness
(`backend/eval.py`) against a customer-supplied set of 25–100 Q&A
cases drawn from their own corpus. The harness records the percentage
of answers that are **grounded** — i.e. every sentence in the answer
has lexical overlap with at least one cited source chunk.

**Our guarantee:**

> **≥ 85% grounded-answer rate on your corpus, audited weekly. If we
> drop below 85% for two consecutive weekly audits, you receive a
> pro-rated refund for the affected month, no questions asked.**

The full methodology — what counts as grounded, how cases are graded,
what triggers the refund, the verifiable code that produces the
numbers — is in [SLA.md](SLA.md). The harness, the per-night CSV
output, and the aggregate summary JSON are all available to the
customer; we don't grade ourselves in private.

No competitor at this price band publishes a measurable quality
guarantee like this. We can because the harness already exists in
this repository (`backend/eval.py`) and the grounding signal it
checks (`citation_fidelity` in `backend/llm.py`) is computed on every
chat response and exposed on the UI as `fidelity_warnings`.

## What is **not** included by default

Separately scoped, separately quoted:

- Bespoke entity-extraction model training on your population
- On-site teacher / SLP / clinician training
- Integrations with proprietary EMR / SIS / LMS systems
- Any work requiring NDA-protected access to your data
- Custom UI white-labelling beyond logo and colour replacement

## Process

1. Email **neilshankarray@vaaani.in** with:
   - Organisation name, country, registration number
   - Use case in 3–5 sentences
   - Estimated number of end-users and seats
   - Required support level (best-effort vs SLA)
2. Indicative quote within 5 working days.
3. If terms work: short Master Services Agreement (Indian Contract Act
   1872) signed; payment is upfront for the first year of any
   subscription. Customer supplies the 25–100 eval cases that anchor
   their SLA; we wire the harness into a nightly cron and share the
   first week's CSV before invoicing starts.
4. You receive a signed commercial-licence letter referencing this
   repository's grant, your name as a permitted commercial user,
   contracted deliverables, and the SLA terms from `SLA.md`.

## Refund and termination

- Subscriptions are otherwise non-refundable after delivery of the
  licence letter and any contracted dashboards / customisations.
- SLA refund (see SLA.md) is the one exception and is automatic.
- Either party may terminate for cause on 30 days' written notice.
- On termination, the patent licence reverts to the free-tier
  permitted-purposes scope.

---

*Questions about which tier you fall into, or how the SLA would
apply to your corpus, just email — easier to ask than to guess wrong.*
