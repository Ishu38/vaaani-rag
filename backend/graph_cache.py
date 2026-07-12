#!/usr/bin/env python3
"""Phase 3 — Precomputed Graph Cache.

Builds a JSON dictionary of every word and root in the graph with its
complete morphological breakdown, phonics mapping, word family, discovery
path, etymology cross-links, and common-error annotations.  The cache is
loaded at startup and enables O(1) deterministic lookups — the graph
router checks the cache before any graph-traversal or LLM call.

Run:  cd backend && python graph_cache.py
Cache is saved to data/graph_cache.json and auto-loaded by the API.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from collections import defaultdict
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import GRAPH_PATH, GRAPH_CACHE_PATH, COMMUNITIES_PATH
from graph import KnowledgeGraph, normalize
from community import load_communities


SPIRAL = pathlib.Path(
    os.environ.get(
        "VAAANI_SPIRAL_PATH",
        os.path.expanduser("~/vaaani-model/dataset/units_spiral.json"),
    )
).expanduser()

# ── Helpers ─────────────────────────────────────────────────────────────────

def _first(items: list, default=""):
    return items[0] if items else default

def _traverse(kg: KnowledgeGraph, src: str, etype: str) -> list[tuple[str, str, dict]]:
    """Return list of (dst_key, display, data) for edges of given type from src."""
    out = []
    for _, dst, d in kg.g.edges(src, data=True):
        if d.get("type") == etype:
            nd = kg.g.nodes.get(dst, {})
            out.append((dst, nd.get("display", dst), d))
    return out

def _incoming(kg: KnowledgeGraph, dst: str, etype: str) -> list[tuple[str, str, dict]]:
    """Return list of (src_key, display, data) for edges of given type into dst."""
    out = []
    for src, _, d in kg.g.in_edges(dst, data=True):
        if d.get("type") == etype:
            nd = kg.g.nodes.get(src, {})
            out.append((src, nd.get("display", src), d))
    return out

def _node_val(kg: KnowledgeGraph, key: str, attr: str, default: Any = ""):
    if not kg.g.has_node(key):
        return default
    return kg.g.nodes[key].get(attr, default)


# ── Build cache ────────────────────────────────────────────────────────────

def build_cache(kg: KnowledgeGraph):
    spiral_data = json.loads(SPIRAL.read_text()) if SPIRAL.exists() else {}
    spiral_roots = spiral_data.get("roots", {})
    spiral_combos = spiral_data.get("_combos", [])

    cache: dict = {
        "words": {},
        "roots": {},
        "phonemes": {},
        "graphemes": {},
        "indexes": {"by_alias": {}, "by_grade": defaultdict(list),
                     "by_root": defaultdict(list), "by_phoneme": defaultdict(list)},
        "stats": {},
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    # ── Pass 1: Build per-word and per-root entries ─────────────────────
    for nid, ndata in kg.g.nodes(data=True):
        ntype = ndata.get("type", "")
        display = ndata.get("display", nid)
        grade = ndata.get("grade")
        descs = ndata.get("descriptions", [])
        gloss = descs[0] if descs else ""

        # ── Word nodes ───────────────────────────────────────────────
        if ntype == "word":
            entry: dict = {
                "display": display, "type": "word",
                "grade": int(grade) if grade is not None else None,
                "gloss": gloss,
                "morphology": {}, "phonics": {},
                "family": [], "discovery": {}, "errors": {},
            }

            # Morphology: root + meaning
            roots = _incoming(kg, nid, "root_of")
            if roots:
                rk, rdisp, _ = roots[0]
                entry["morphology"]["root"] = rdisp
                entry["morphology"]["root_key"] = rk
                meanings = _traverse(kg, rk, "means")
                if meanings:
                    entry["morphology"]["meaning"] = meanings[0][1]

            # Phonics: first phoneme
            phonemes = _traverse(kg, nid, "sounds_like")
            if phonemes:
                pk, pdisp, _ = phonemes[0]
                entry["phonics"]["phoneme"] = pdisp
                entry["phonics"]["phoneme_key"] = pk
                graphemes = _traverse(kg, pk, "written_as")
                entry["phonics"]["graphemes"] = [gd for _, gd, _ in graphemes]
                cache["indexes"]["by_phoneme"][pk].append(nid)

            # Family: siblings from root
            if roots:
                rk = roots[0][0]
                for _, wd, d in _traverse(kg, rk, "root_of"):
                    if wd not in entry["family"]:
                        entry["family"].append(wd)
                entry["family"] = sorted(entry["family"])
                cache["indexes"]["by_root"][rk].append(nid)

            # Discovery: next words
            next_w = _traverse(kg, nid, "prerequisite_for")
            prev_w = _incoming(kg, nid, "prerequisite_for")
            entry["discovery"]["next"] = [nd for _, nd, _ in next_w[:6]]
            entry["discovery"]["prev"] = [nd for _, nd, _ in prev_w[:3]]
            if entry["discovery"]["next"]:
                entry["discovery"]["ready"] = f"Now that you know {display}, you can discover: {', '.join(entry['discovery']['next'][:4])}."

            # Errors: inherits from spiral fakes
            for rkey, rdata in spiral_roots.items():
                piece_lower = rdata.get("piece", "").lower()
                if entry.get("morphology", {}).get("root", "").lower() == piece_lower:
                    fakes = rdata.get("fakes", [])
                    if fakes:
                        entry["errors"]["fakes"] = fakes
                        entry["errors"]["watch_for"] = (
                            f"Watch out! These words look like they belong to the "
                            f"{rdata['piece']} family but they don't: {', '.join(fakes)}."
                        )

            # Grade index
            if entry["grade"]:
                cache["indexes"]["by_grade"][str(entry["grade"])].append(nid)

            cache["words"][normalize(display)] = entry

        # ── Root nodes ─────────────────────────────────────────────────
        elif ntype == "root":
            entry: dict = {
                "display": display, "type": "root",
                "grade": int(grade) if grade is not None else None,
                "meaning": "", "words": [], "fakes": [],
                "family_ready": "",
            }

            meanings = _traverse(kg, nid, "means")
            if meanings:
                entry["meaning"] = meanings[0][1]

            # All words using this root
            words = _traverse(kg, nid, "root_of")
            entry["words"] = sorted([wd for _, wd, _ in words])

            # Fakes from spiral
            for rkey, rdata in spiral_roots.items():
                if rdata.get("piece", "").lower() == display.lower():
                    entry["fakes"] = rdata.get("fakes", [])
                    entry["meaning"] = rdata.get("meaning", entry["meaning"])
                    entry["story"] = rdata.get("story", [""])[0] if rdata.get("story") else ""
                    entry["char"] = rdata.get("char", "")
                    entry["emoji"] = rdata.get("emoji", "")
                    entry["conn"] = rdata.get("conn", {})

            # Etymology links
            lang_links = {}
            for _, dst, d in kg.g.edges(nid, data=True):
                etype = d.get("type", "")
                if etype in ("cognate_with", "translates_to"):
                    lang_links[d.get("descriptions", [""])[0]] = _node_val(kg, dst, "display", dst)
            entry["etymology_links"] = lang_links

            if entry["words"]:
                entry["family_ready"] = (
                    f"The {display} family: {', '.join(entry['words'][:8])}. "
                    f"Each word carries the idea of '{entry['meaning']}'."
                )

            cache["roots"][normalize(display)] = entry

        # ── Phoneme nodes ────────────────────────────────────────────────
        elif ntype == "phoneme":
            entry = {
                "display": display,
                "voice": ndata.get("voice"),
                "place": ndata.get("place", ""),
                "manner": ndata.get("manner", ""),
                "examples": ndata.get("examples", [])[:8],
                "aliases": [],
            }
            # Collect aliases
            for src, _, d in kg.g.in_edges(nid, data=True):
                if d.get("type") == "alias_of":
                    alias_display = _node_val(kg, src, "display", src)
                    entry["aliases"].append(alias_display)
                    cache["indexes"]["by_alias"][normalize(alias_display)] = nid
            # Collect graphemes
            entry["graphemes"] = [gd for _, gd, _ in _traverse(kg, nid, "written_as")]
            cache["phonemes"][nid] = entry

        # ── Grapheme nodes ────────────────────────────────────────────────
        elif ntype == "grapheme":
            entry = {"display": display}
            entry["phonemes"] = [pd for _, pd, _ in _traverse(kg, nid, "sounds_like")]
            cache["graphemes"][nid] = entry

    # ── Pass 2: Combo words that have multiple roots ─────────────────────
    for combo in spiral_combos:
        w = normalize(combo["w"])
        if w in cache["words"]:
            entry = cache["words"][w]
            entry["morphology"]["combo"] = True
            entry["morphology"]["roots"] = [combo["a"], combo["b"]]
            entry["morphology"]["combo_gloss"] = combo.get("g", "")

    # ── Pass 3: Reversal 1 — Prerequisite dependency chains ───────────────
    # For each word, find which words must be known FIRST (reverse of
    # prerequisite_for edges). This enables gap analysis: "To learn X,
    # you need Y → Z → ..."
    from collections import deque

    # Build reverse index: word → [words that prerequisite into this one]
    reverse_prereqs: dict[str, list[str]] = defaultdict(list)
    for word_key, wdata in cache["words"].items():
        next_words = wdata.get("discovery", {}).get("next", [])
        for nw in next_words:
            nw_n = normalize(nw)
            reverse_prereqs[nw_n].append(word_key)

    # Walk prerequisite chains (up to depth 3) for every word
    for word_key, wdata in cache["words"].items():
        prereq_chain: list[str] = []
        prereq_display: list[str] = []
        visited: set[str] = set()
        queue = deque(reverse_prereqs.get(word_key, [])[:4])
        depth = 0
        while queue and depth < 3:
            for _ in range(len(queue)):
                pw = queue.popleft()
                if pw in visited or pw not in cache["words"]:
                    continue
                visited.add(pw)
                pdata = cache["words"][pw]
                prereq_chain.append(pw)
                prereq_display.append(pdata.get("display", pw))
                # Walk one level deeper
                for gp in reverse_prereqs.get(pw, [])[:2]:
                    if gp not in visited:
                        queue.append(gp)
            depth += 1
        wdata["prerequisites"] = prereq_chain[:6]
        wdata["prerequisite_display"] = prereq_display[:6]
        if prereq_display:
            wdata["prerequisite_chain"] = (
                f"Before you can learn {wdata.get('display', word_key)}, "
                f"you should know: {' → '.join(prereq_display[:4])}."
            )

    # ── Stats ────────────────────────────────────────────────────────────
    cache["stats"]["total_words"] = len(cache["words"])
    cache["stats"]["total_roots"] = len(cache["roots"])
    cache["stats"]["total_phonemes"] = len(cache["phonemes"])
    cache["stats"]["total_graphemes"] = len(cache["graphemes"])
    cache["stats"]["total_aliases"] = len(cache["indexes"]["by_alias"])
    cache["stats"]["deterministic_coverage"] = (
        f"{cache['stats']['total_words']} words answerable without LLM — "
        f"covering {cache['stats']['total_roots']} root families"
    )

    return cache


# ── CLI & load ──────────────────────────────────────────────────────────────

def rebuild_and_save():
    kg = KnowledgeGraph.load(GRAPH_PATH)
    cache = build_cache(kg)
    GRAPH_CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    print(f"Cache built: {cache['stats']['total_words']} words, "
          f"{cache['stats']['total_roots']} roots, "
          f"{cache['stats']['total_phonemes']} phonemes")
    print(f"Saved → {GRAPH_CACHE_PATH}")
    return cache


_loaded_cache: dict | None = None


def load_cache() -> dict:
    global _loaded_cache
    if _loaded_cache is not None:
        return _loaded_cache
    if GRAPH_CACHE_PATH.exists():
        _loaded_cache = json.loads(GRAPH_CACHE_PATH.read_text())
    else:
        _loaded_cache = {}
    return _loaded_cache


if __name__ == "__main__":
    rebuild_and_save()
