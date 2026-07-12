#!/usr/bin/env python3
"""Build the Vaaani Core Library — the curated curriculum a learner explores
WITHOUT uploading anything (the B2C young-learner face of the two-audience,
one-engine design).

It turns the vetted grade-spiral roots (vaaani-model/dataset/units_spiral.json)
into entity-rich "world" documents, ingests them through the normal pipeline
(so they are answerable in chat AND become stars in the graph), and marks each
as Library content — readable by every learner. A brand-new user then opens a
populated universe instead of an empty map.

Run (backend on the host, engine on :8011 for graph extraction):
    cd backend && ../.venv/bin/python library/build_library.py            # grade 1
    ../.venv/bin/python library/build_library.py --grades 1 2 3           # more
    ../.venv/bin/python library/build_library.py --vectors-only           # fast, no graph

Idempotent: re-running re-ingests changed docs and re-marks ownership.
"""
import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import INDEX_PATH, METADATA_PATH, RAW_DIR
import scope

# Curriculum source. Overridable so the same script seeds on the deployed VM
# (VAAANI_SPIRAL_PATH=~/rag-assistant/backend/data/units_spiral.json) without
# editing code; default keeps local-laptop dev working unchanged.
SPIRAL = pathlib.Path(
    os.environ.get("VAAANI_SPIRAL_PATH", "/home/ishu/vaaani-model/dataset/units_spiral.json")
).expanduser()
LIB_DIR = RAW_DIR / "library"

# Grade-1 roots grouped into explorable "worlds" (the vision's Library worlds).
# Each world → one document. Roots not listed fall into "More Word Roots".
WORLDS = {
    "Numbers World": ["tri", "uni", "bi", "octo", "cent", "nov"],
    "Body and People World": ["dent", "ped", "manu"],
    "Nature and Sky World": ["aqua", "sol", "luna", "terr", "flor", "zoo"],
}


def _conn_sentence(conn: dict, piece: str, meaning: str) -> str:
    """Honest home-language bridge: a cognate is a true shared-ancestor cousin;
    a translation is only the home word for the idea (never claimed as kin).
    Matches the integrity rule baked into the curriculum data."""
    hi, bn = conn.get("hi", ""), conn.get("bn", "")
    if conn.get("type") == "cognate":
        src = conn.get("src", "")
        return (f"Long ago, {piece} and the Indian word {src} were the same word — "
                f"they are true cousins. In Hindi we say {hi}, and in Bengali {bn}, "
                f"for '{meaning}'.")
    return (f"In Hindi, the word for '{meaning}' is {hi}, and in Bengali it is {bn}. "
            f"(These are your home words for the idea, not cousins of {piece}.)")


def _root_prose(key: str, r: dict) -> str:
    piece, meaning, char = r["piece"], r["meaning"], r["char"]
    fam = r.get("family", [])
    new = r.get("new", [])
    lines = [f"The word root {piece} means '{meaning}'."]
    lines.append(" ".join(s for s in r.get("story", [])))
    if fam:
        fam_list = ", ".join(f"{w['w']} ({w['g']})" for w in fam)
        fam_words = ", ".join(w["w"] for w in fam)
        lines.append(f"The word family of {piece} includes {fam_list}. "
                     f"The words {fam_words} all share the hidden piece {piece}, "
                     f"which is why they all carry the meaning '{meaning}'.")
    if new:
        new_list = ", ".join(f"{w['w']} ({w['g']})" for w in new)
        lines.append(f"Two more words in the {piece} family are {new_list}.")
    lines.append(_conn_sentence(r.get("conn", {}), piece, meaning))
    return " ".join(lines)


def build_docs(roots: dict, grades: list[int]) -> list[pathlib.Path]:
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    active = {k: v for k, v in roots.items() if v["grade"] in grades}
    assigned = {k for ks in WORLDS.values() for k in ks if k in active}
    worlds = dict(WORLDS)
    leftovers = [k for k in active if k not in assigned]
    if leftovers:
        worlds["More Word Roots"] = leftovers

    written: list[pathlib.Path] = []
    for world, keys in worlds.items():
        keys = [k for k in keys if k in active]
        if not keys:
            continue
        parts = [f"# {world}\n",
                 f"Welcome to the {world}! Here we discover hidden word roots — "
                 f"the little shared pieces inside many words.\n"]
        for k in keys:
            r = active[k]
            parts.append(f"## {r['char']} and the root {r['piece']}\n")
            parts.append(_root_prose(k, r) + "\n")
        fname = "library-" + world.lower().replace(" ", "-") + ".md"
        path = LIB_DIR / fname
        path.write_text("\n".join(parts), encoding="utf-8")
        written.append(path)
        print(f"  wrote {fname} ({len(keys)} roots)")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grades", type=int, nargs="+", default=[1])
    ap.add_argument("--vectors-only", action="store_true",
                    help="skip the slow LLM graph extraction (chat grounding only)")
    args = ap.parse_args()

    roots = json.loads(SPIRAL.read_text())["roots"]
    print(f"Building Core Library for grades {args.grades}…")
    docs = build_docs(roots, args.grades)
    if not docs:
        print("No roots for those grades."); return

    # Ingest the whole library dir through the normal pipeline.
    from ingest import ingest
    print("Ingesting (this runs graph extraction unless --vectors-only)…")
    summary = ingest(LIB_DIR, INDEX_PATH, METADATA_PATH, build_graph=not args.vectors_only)
    print(f"  chunks_added={summary.get('chunks_added')} total={summary.get('total_chunks')}")

    # Mark every library doc as readable by all learners.
    for p in docs:
        scope.record_library_ownership(str(p.resolve()))
    print(f"  marked {len(docs)} docs as Core Library (visible to every learner)")

    try:
        from main import retriever
        retriever.reload()
        print("  retriever reloaded")
    except Exception as e:
        print(f"  (reload skipped: {e} — restart the backend to pick it up)")


if __name__ == "__main__":
    main()
