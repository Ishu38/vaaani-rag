"""Λ_t projection — the Linguistic Cognitive State of one learner.

Research instrument for LINGUISTIC_STATE_THEORY.md §1: reads the domain
graph + the cognitive twin DB and prints the learner's belief field
projected onto linguistic tiers (phonology–orthography, morphology,
cross-linguistic, lexical–semantic), with evidence counts and coverage.
Every number is a deterministic view of the evidence set (Part II, A5).

Usage:
    python scripts/lambda_state.py <student_id> [--json]
    python scripts/lambda_state.py --list          # students with evidence
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import cognitive_twin as twin                      # noqa: E402
from evidence_graph import DB_PATH                 # noqa: E402
from config import GRAPH_PATH                      # noqa: E402

TIERS = ("phonology-orthography", "morphology", "cross-linguistic",
         "lexical-semantic", "untyped")


def tier_partition(graph_path: Path = GRAPH_PATH) -> dict[str, str]:
    """node_id -> tier, derived from node `type` + incident edge types.
    Mirrors LINGUISTIC_STATE_THEORY.md §1.1 exactly."""
    g = json.loads(Path(graph_path).read_text())
    incident: dict[str, set] = defaultdict(set)
    for e in g["links"]:
        incident[e["source"]].add(e.get("type"))
        incident[e["target"]].add(e.get("type"))

    out: dict[str, str] = {}
    for n in g["nodes"]:
        t, nid, ed = n.get("type", "unknown"), n["id"], incident[n["id"]]
        if t in ("phoneme", "grapheme") or "sounds_like" in ed or "written_as" in ed:
            out[nid] = "phonology-orthography"
        elif t in ("root", "prefix", "suffix") or "root_of" in ed or "word_family" in ed:
            out[nid] = "morphology"
        elif "translates_to" in ed or "cognate_with" in ed:
            out[nid] = "cross-linguistic"
        elif t in ("word", "meaning", "language-word") or "means" in ed:
            out[nid] = "lexical-semantic"
        else:
            out[nid] = "untyped"
    return out


def lambda_state(student_id: str) -> dict:
    import credal
    tiers = tier_partition()
    beliefs = twin.snapshot(student_id)
    credal_snap = credal.snapshot(student_id)

    with sqlite3.connect(DB_PATH) as c:
        ev = dict(c.execute(
            "SELECT node_id, COUNT(*) FROM evidence WHERE student_id=? "
            "GROUP BY node_id", (student_id,)).fetchall())

    proj: dict[str, dict] = {t: {"nodes_total": 0, "nodes_met": 0,
                                 "mastery_mean": 0.0, "mastered": 0,
                                 "evidence": 0, "uncertainty_sd": None,
                                 "_sum": 0.0, "_sd_sum": 0.0, "_sd_n": 0}
                             for t in TIERS}
    for nid, t in tiers.items():
        proj[t]["nodes_total"] += 1
    for nid, b in beliefs.items():
        t = tiers.get(nid)
        if t is None:            # twin row for a node no longer in the graph
            continue
        p = proj[t]
        p["nodes_met"] += 1
        p["_sum"] += b.mastery
        p["mastered"] += int(b.mastered)
        p["evidence"] += ev.get(nid, 0)
        if nid in credal_snap:
            p["_sd_sum"] += credal_snap[nid].sd
            p["_sd_n"] += 1
    for t, p in proj.items():
        if p["nodes_met"]:
            p["mastery_mean"] = round(p["_sum"] / p["nodes_met"], 4)
        if p["_sd_n"]:
            p["uncertainty_sd"] = round(p["_sd_sum"] / p["_sd_n"], 4)
        p["coverage"] = round(p["nodes_met"] / p["nodes_total"], 4) if p["nodes_total"] else 0.0
        del p["_sum"], p["_sd_sum"], p["_sd_n"]

    return {"student_id": student_id,
            "evidence_total": sum(ev.values()),
            "nodes_met": len(beliefs),
            "lambda": proj}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("student_id", nargs="?")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        with sqlite3.connect(DB_PATH) as c:
            for sid, n in c.execute(
                    "SELECT student_id, COUNT(*) FROM evidence "
                    "GROUP BY student_id ORDER BY 2 DESC"):
                print(f"{sid:24s} {n} evidence objects")
        return 0
    if not args.student_id:
        ap.error("student_id required (or --list)")

    lam = lambda_state(args.student_id)
    if args.json:
        print(json.dumps(lam, indent=2))
        return 0

    print(f"Λ_t for {lam['student_id']} — {lam['evidence_total']} evidence "
          f"objects over {lam['nodes_met']} nodes\n")
    print(f"{'tier':24s} {'met/total':>10s} {'coverage':>9s} "
          f"{'mastery':>8s} {'mastered':>9s} {'evidence':>9s} {'±sd':>7s}")
    for t in TIERS:
        p = lam["lambda"][t]
        sd = f"{p['uncertainty_sd']:.3f}" if p["uncertainty_sd"] is not None else "—"
        print(f"{t:24s} {p['nodes_met']:>4d}/{p['nodes_total']:<5d} "
              f"{p['coverage']:>9.1%} {p['mastery_mean']:>8.3f} "
              f"{p['mastered']:>9d} {p['evidence']:>9d} {sd:>7s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
