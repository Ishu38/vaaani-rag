"""Prerequisite enrichment — derive pedagogically-ordered edges from the
linguistic edge types already in the graph, without touching graph.json.

Rules (linguistically defensible only; no statistical guessing):
  root_of:  the root is prerequisite for each derived word (you can't analyze
            'unhappiness' before knowing 'happy' — Kurdi ch. 3, morphological
            decomposition presupposes the base)
  is_a:     the parent category is prerequisite for the specific member

Output: data/prereq_overlay.json  — loaded by development_engine.WorldModel
as an overlay (same revertible-overrides discipline as CAVP's
attractor_overrides.json: fitted/derived knowledge never overwrites source
data, and rolling back = deleting one file).

Run:  python scripts/enrich_prereqs.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRAPH = ROOT / "data" / "graph.json"
OUT = ROOT / "data" / "prereq_overlay.json"


def main() -> None:
    g = json.loads(GRAPH.read_text())
    node_ids = {n["id"] for n in g["nodes"]}
    existing = set()
    for e in g["links"]:
        if e.get("type") in ("prerequisite_for", "depends_on"):
            existing.add((e["source"], e["target"]))

    derived: list[dict] = []

    def add(pre: str, post: str, rule: str) -> None:
        if pre in node_ids and post in node_ids and pre != post:
            if (pre, post) not in existing:
                derived.append({"prerequisite": pre, "for": post, "rule": rule})

    for e in g["links"]:
        t = e.get("type")
        if t == "root_of":
            # root_of: source is the root of target
            add(e["source"], e["target"], "root_of")
        elif t == "is_a":
            # X is_a Y: parent concept Y before specific X
            add(e["target"], e["source"], "is_a")

    # de-duplicate
    seen, unique = set(), []
    for d in derived:
        k = (d["prerequisite"], d["for"])
        if k not in seen:
            seen.add(k)
            unique.append(d)

    OUT.write_text(json.dumps({
        "generated": "scripts/enrich_prereqs.py",
        "rules": {"root_of": "root before derived word", "is_a": "category before member"},
        "edges": unique,
    }, indent=1))
    print(f"prereq overlay: {len(unique)} derived edges -> {OUT}")


if __name__ == "__main__":
    main()
