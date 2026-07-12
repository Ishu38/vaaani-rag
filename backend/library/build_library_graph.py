#!/usr/bin/env python3
"""Build the Core Library's constellation DETERMINISTICALLY from the vetted
curriculum (units_spiral.json) — no LLM, no engine, instant, curated.

For a hand-authored library this beats LLM extraction on every axis: the exact
stars we want (a root piece + its family words), correct global chunk
attribution (so per-user visibility works), and each word family becomes its
own constellation — literally the vision's "Word Families -> Constellations".

Run:  cd backend && ../.venv/bin/python library/build_library_graph.py
Then restart the backend to load the universe.
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import GRAPH_PATH, COMMUNITIES_PATH, METADATA_PATH
from graph import KnowledgeGraph, normalize
from extractor import Entity, Relation
from community import Community, load_communities, save_communities

# See build_library.py: overridable so the deployed VM can point at its own copy.
SPIRAL = pathlib.Path(
    os.environ.get("VAAANI_SPIRAL_PATH", "/home/ishu/vaaani-model/dataset/units_spiral.json")
).expanduser()
GRADES = [int(g) for g in (sys.argv[1:] or ["1"])]

# world doc filename each root was written into (mirrors build_library.WORLDS)
WORLD_FILE = {}
for k in ["tri", "uni", "bi", "octo", "cent", "nov"]:
    WORLD_FILE[k] = "library-numbers-world.md"
for k in ["dent", "ped", "manu"]:
    WORLD_FILE[k] = "library-body-and-people-world.md"
for k in ["aqua", "sol", "luna", "terr", "flor", "zoo"]:
    WORLD_FILE[k] = "library-nature-and-sky-world.md"


def main() -> None:
    roots = json.loads(SPIRAL.read_text())["roots"]
    meta = json.loads(METADATA_PATH.read_text())
    # first global chunk id per library source filename (any chunk of that file
    # shares the library path, which is all node_visible needs).
    chunk_for_source: dict[str, int] = {}
    for i, c in enumerate(meta.get("chunks", [])):
        src = c.get("source", "")
        if "/library/" in c.get("path", "") and src not in chunk_for_source:
            chunk_for_source[src] = i
    if not chunk_for_source:
        print("No library chunks — run build_library.py first."); return

    kg = KnowledgeGraph.load(GRAPH_PATH)
    before = kg.g.number_of_nodes()

    existing = load_communities(COMMUNITIES_PATH)
    next_cid = (max((c.id for c in existing), default=-1)) + 1
    new_comms: list[Community] = []

    n_roots = 0
    for key, r in roots.items():
        if r["grade"] not in GRADES:
            continue
        fname = WORLD_FILE.get(key)
        cid_chunk = chunk_for_source.get(fname)
        if cid_chunk is None:
            continue
        piece, meaning = r["piece"], r["meaning"]
        root_key = normalize(piece)                       # e.g. "tri"
        # root piece star
        kg.add_entity(Entity(name=piece, type="root",
                             description=f"a word root meaning '{meaning}'"), cid_chunk)
        member_keys = [root_key]
        # family + new words as stars, each linked to the root
        for w in (r.get("family", []) + r.get("new", [])):
            word, gloss = w["w"], w["g"]
            kg.add_entity(Entity(name=word, type="word", description=gloss), cid_chunk)
            kg.add_relation(Relation(source=piece, target=word, type="root_of",
                                    description=f"{word} carries the root {piece} ({meaning})"),
                            cid_chunk)
            member_keys.append(normalize(word))

        # each word family is a constellation
        fam_words = [w["w"] for w in r.get("family", [])]
        new_comms.append(Community(
            id=next_cid,
            nodes=member_keys,
            title=f"The {piece} family — {meaning}",
            summary=(r.get("story", [""])[0] + f" The root {piece} means '{meaning}'."),
            findings=[f"{w['w']}: {w['g']}" for w in r.get("family", [])],
            size=len(member_keys),
        ))
        next_cid += 1
        n_roots += 1

    kg.save(GRAPH_PATH)
    save_communities(existing + new_comms, COMMUNITIES_PATH)
    print(f"Library graph: {before} -> {kg.g.number_of_nodes()} nodes "
          f"(+{n_roots} root families, +{len(new_comms)} constellations)")

    # Phase 1 — structural linguistics enrichment
    print("\n--- Phase 1: structural linguistics enrichment ---")
    from graph_seeder import seed as seed_linguistics
    lingo_stats = seed_linguistics(kg)
    kg.save(GRAPH_PATH)
    print(f"After enrichment: {kg.g.number_of_nodes()} nodes, "
          f"{kg.g.number_of_edges()} edges")
    for k, v in sorted(lingo_stats.items()):
        print(f"  {k}: {v}")
    new_communities_count = lingo_stats.get("new_communities", 0)
    print(f"  (+{new_communities_count} new communities)")

    print("Done. Restart the backend to load the universe.")


if __name__ == "__main__":
    main()
