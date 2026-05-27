"""Vaaani RAG quality eval harness — answers honest questions about quality.

Run this against a live brain.vaaani.in (or local dev server) with a small
set of hand-graded Q&A cases. It records per-case scores and an aggregate
report you can track over time to spot regressions.

What gets measured per case:

  1. ``grounded``     — did the answer come back with zero fidelity_warnings?
                        (warnings ⇒ at least one sentence with <20% lexical
                        overlap with retrieved chunks)
  2. ``source_hit``   — if the case specifies ``expected_source_substring``,
                        was at least one returned source's filename a match?
  3. ``mentions_all`` — if the case specifies ``expected_terms`` (e.g.
                        ["phonogram", "Rule 17"]), does the answer contain
                        ALL of them (case-insensitive)?
  4. ``intent``       — does the classified intent match the case's
                        ``expected_intent`` (if given)?
  5. ``latency_ms``   — total wall-clock time of the /chat call

Usage:
    python3 backend/eval.py \
        --base-url https://brain.vaaani.in \
        --session-cookie VAAANI_SESSION=... \
        --cases backend/eval_cases.json \
        --out runs/eval_$(date +%F).csv

The session cookie is required because /chat returns zeros for anonymous
callers. Get it from your browser's devtools after signing in.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _post_chat(base_url: str, cookie: str | None, query: str, timeout: float = 60) -> dict:
    body = json.dumps({
        "query": query,
        "conversation_history": [],
        "socratic": False,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if cookie:
        headers["Cookie"] = cookie
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat",
        data=body, headers=headers, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _score_case(case: dict, response: dict, latency_ms: int) -> dict:
    answer = (response.get("answer") or "").lower()
    sources = response.get("sources") or []
    src_names = [s.get("source", "") for s in sources]
    fidelity_warnings = response.get("fidelity_warnings") or []
    intent_got = response.get("intent")

    grounded = 1 if not fidelity_warnings else 0

    src_sub = case.get("expected_source_substring")
    if src_sub:
        source_hit = 1 if any(src_sub.lower() in s.lower() for s in src_names) else 0
    else:
        source_hit = None  # not graded

    terms = case.get("expected_terms") or []
    if terms:
        mentions_all = 1 if all(t.lower() in answer for t in terms) else 0
        missing = [t for t in terms if t.lower() not in answer]
    else:
        mentions_all = None
        missing = []

    exp_intent = case.get("expected_intent")
    intent_match = 1 if (exp_intent is None or exp_intent == intent_got) else 0

    return {
        "query": case.get("query", ""),
        "grounded": grounded,
        "source_hit": source_hit,
        "mentions_all": mentions_all,
        "missing_terms": ", ".join(missing),
        "intent_expected": exp_intent or "",
        "intent_got": intent_got or "",
        "intent_match": intent_match,
        "fidelity_warning_count": len(fidelity_warnings),
        "source_count": len(src_names),
        "latency_ms": latency_ms,
    }


def _summarise(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}

    def _mean_pct(key: str) -> float:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return float("nan")
        return 100.0 * sum(vals) / len(vals)

    lat = [r["latency_ms"] for r in rows]
    lat_sorted = sorted(lat)
    p50 = lat_sorted[len(lat_sorted) // 2]
    p95 = lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))]
    return {
        "n": n,
        "grounded_pct": _mean_pct("grounded"),
        "source_hit_pct": _mean_pct("source_hit"),
        "mentions_all_pct": _mean_pct("mentions_all"),
        "intent_match_pct": _mean_pct("intent_match"),
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Vaaani RAG eval harness")
    p.add_argument("--base-url", default="https://brain.vaaani.in",
                   help="Backend root URL")
    p.add_argument("--session-cookie", default="",
                   help='Full cookie string, e.g. "vaaani_session=eyJ..."')
    p.add_argument("--cases", default=str(Path(__file__).parent / "eval_cases.json"),
                   help="Path to JSON file with the eval cases")
    p.add_argument("--out", default="",
                   help="Optional CSV output path for per-case rows")
    p.add_argument("--timeout", type=float, default=60.0,
                   help="Per-call timeout seconds (default 60)")
    args = p.parse_args()

    try:
        cases = json.loads(Path(args.cases).read_text())
    except FileNotFoundError:
        print(f"[err] cases file not found: {args.cases}", file=sys.stderr)
        return 2
    if not isinstance(cases, list):
        print("[err] cases file must be a JSON array of case objects", file=sys.stderr)
        return 2

    rows: list[dict] = []
    failures = 0
    print(f"running {len(cases)} cases against {args.base_url}")
    for i, case in enumerate(cases, 1):
        q = case.get("query")
        if not q:
            print(f"  [{i}/{len(cases)}] skipped: case missing 'query'")
            continue
        t0 = time.monotonic()
        try:
            response = _post_chat(args.base_url, args.session_cookie or None, q, timeout=args.timeout)
        except urllib.error.HTTPError as e:
            print(f"  [{i}/{len(cases)}] HTTP {e.code}: {q[:60]}")
            failures += 1
            continue
        except Exception as e:
            print(f"  [{i}/{len(cases)}] error: {e}: {q[:60]}")
            failures += 1
            continue
        latency_ms = int((time.monotonic() - t0) * 1000)
        row = _score_case(case, response, latency_ms)
        rows.append(row)
        marks = []
        if row["grounded"]:                     marks.append("grounded")
        if row.get("source_hit") == 1:          marks.append("source-hit")
        if row.get("mentions_all") == 1:        marks.append("mentions-all")
        if row["intent_match"]:                 marks.append("intent-OK")
        print(f"  [{i}/{len(cases)}] {latency_ms:>5}ms  {','.join(marks) or '—'}  | {q[:70]}")

    summary = _summarise(rows)
    print("\n──────────── SUMMARY ────────────")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:22} {v:.1f}")
        else:
            print(f"  {k:22} {v}")
    if failures:
        print(f"  network/HTTP failures   {failures}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as fh:
            if rows:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        # Append summary as JSON sidecar — easy to chart with pandas later.
        Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nwrote per-case CSV: {out_path}")
        print(f"wrote summary JSON: {Path(args.out).with_suffix('.summary.json')}")

    # Exit non-zero if any case scored poorly — useful in CI/cron.
    bad = sum(
        1 for r in rows
        if (r["grounded"] == 0)
        or (r.get("source_hit") == 0)
        or (r.get("mentions_all") == 0)
        or (r["intent_match"] == 0)
    )
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
