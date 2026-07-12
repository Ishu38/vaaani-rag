#!/usr/bin/env python3
"""Build graph stars for the Core Library docs — correctly and safely.

Why this exists instead of the normal graph phase:
  1. The deferred graph builder attributes nodes with FILE-LOCAL chunk ids,
     which would map Library stars onto the wrong file's path and break the
     per-user visibility scope. Here we use the true GLOBAL chunk index.
  2. On the slow CPU engine, batched extraction was timing out and silently
     leaving zero nodes. Here each chunk is one sequential call, comfortably
     under DEEPSEEK_TIMEOUT.

Run (engine on :8011):  cd backend && ../.venv/bin/python library/rebuild_library_graph.py
Then restart the backend so it reloads the graph.
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import GRAPH_PATH, COMMUNITIES_PATH, METADATA_PATH
from graph import KnowledgeGraph
from extractor import extract_chunk
from community import build_communities, save_communities


def main() -> None:
    meta = json.loads(METADATA_PATH.read_text())
    chunks = meta.get("chunks", [])
    lib_idx = [i for i, c in enumerate(chunks) if "/library/" in c.get("path", "")]
    if not lib_idx:
        print("No library chunks found — run build_library.py first."); return
    print(f"Extracting stars for {len(lib_idx)} Library chunks (global ids: {lib_idx})…")

    kg = KnowledgeGraph.load(GRAPH_PATH)
    before = kg.g.number_of_nodes()
    added_e = added_r = 0
    for gid in lib_idx:
        text = chunks[gid]["text"]
        t0 = time.time()
        try:
            ex = extract_chunk(text)                 # global chunk id below
        except Exception as e:
            print(f"  chunk {gid}: extract failed ({e}) — skipping"); continue
        kg.ingest_extraction(ex, gid)                # CORRECT global attribution
        added_e += len(ex.entities); added_r += len(ex.relations)
        print(f"  chunk {gid}: +{len(ex.entities)} entities, "
              f"+{len(ex.relations)} relations ({time.time()-t0:.0f}s)")

    kg.save(GRAPH_PATH)
    print(f"Graph: {before} -> {kg.g.number_of_nodes()} nodes "
          f"(+{added_e} entities, +{added_r} relations)")

    if kg.g.number_of_nodes() > 0:
        print("Rebuilding communities (constellations)…")
        comms = build_communities(kg)
        save_communities(comms, COMMUNITIES_PATH)
        print(f"  {len(comms)} constellations")
    print("Done. Restart the backend to load the new universe.")


if __name__ == "__main__":
    main()
