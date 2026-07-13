#!/usr/bin/env python3
"""Phase 5 — Curriculum Graph Loader.

Loads curriculum-to-graph-node mappings from structured JSON files so
schools can plug in their own standards without touching the Language
Graph. The curriculum maps grade-level outcomes to specific graph nodes
(roots, words, phonemes) from the Language Graph, enabling:

  - Grade-aware discovery missions (only suggest words at the learner's level)
  - Curriculum-constrained chat (recommend concepts the school requires)
  - Progress tracking (which standards has the learner met)
  - Multi-curriculum support (NCERT, CBSE, Cambridge — one Language Graph)

Curriculum files live in data/curriculum_*.json
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from collections import defaultdict
from typing import Any, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import DATA_DIR
from graph import normalize


CURRICULUM_DIR = DATA_DIR


def _available_curricula() -> list[str]:
    """Find all curriculum JSON files in the data directory."""
    out = []
    for f in CURRICULUM_DIR.glob("curriculum_*.json"):
        out.append(f.stem.replace("curriculum_", ""))
    return sorted(out)


def load_curriculum(name: str) -> dict:
    """Load a specific curriculum by name (ncert, cbse, cambridge)."""
    path = CURRICULUM_DIR / f"curriculum_{name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def words_for_grade(name: str, grade: int) -> list[str]:
    """Return all words required by a curriculum at a given grade."""
    curr = load_curriculum(name)
    grade_str = str(grade)
    return curr.get("grades", {}).get(grade_str, {}).get("required_words", [])


def roots_for_grade(name: str, grade: int) -> list[str]:
    """Return all roots required by a curriculum at a given grade."""
    curr = load_curriculum(name)
    grade_str = str(grade)
    return curr.get("grades", {}).get(grade_str, {}).get("required_roots", [])


def phonics_for_grade(name: str, grade: int) -> list[str]:
    """Return phonics focus sounds for a given grade."""
    curr = load_curriculum(name)
    grade_str = str(grade)
    return curr.get("grades", {}).get(grade_str, {}).get("phonics_focus", [])


def outcomes_for_grade(name: str, grade: int) -> list[str]:
    """Return learning outcomes for a given grade."""
    curr = load_curriculum(name)
    grade_str = str(grade)
    return curr.get("grades", {}).get(grade_str, {}).get("outcomes", [])


def is_required(name: str, grade: int, word: str) -> bool:
    """Check if a word is required by a curriculum at a given grade."""
    words = words_for_grade(name, grade)
    return normalize(word) in [normalize(w) for w in words]


def grade_for_word(name: str, word: str) -> Optional[int]:
    """Find the earliest grade that requires a given word in the curriculum."""
    curr = load_curriculum(name)
    wn = normalize(word)
    for g in sorted(int(k) for k in curr.get("grades", {}).keys()):
        words = curr["grades"][str(g)].get("required_words", [])
        if wn in [normalize(w) for w in words]:
            return g
    return None


def available_words_at(name: str, grade: int) -> list[str]:
    """All words the learner has access to at this grade (cumulative: grade and below)."""
    curr = load_curriculum(name)
    all_words = []
    for g in sorted(int(k) for k in curr.get("grades", {}).keys()):
        if g <= grade:
            all_words.extend(curr["grades"][str(g)].get("required_words", []))
    return all_words


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Available curricula: {_available_curricula()}")
    for name in _available_curricula():
        c = load_curriculum(name)
        print(f"\n{name.upper()}:")
        for g in sorted(c.get("grades", {}).keys(), key=int):
            gd = c["grades"][g]
            roots = gd.get("required_roots", [])
            words = gd.get("required_words", [])
            print(f"  Grade {g}: {len(roots)} roots, {len(words)} words — {gd.get('label', '')}")
