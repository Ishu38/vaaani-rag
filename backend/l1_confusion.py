"""Contrastive L1 confusion edges — inhibitory CASCADE links.

Contrastive Analysis (Lado 1957) + the Speech Learning Model (Flege) predict
that an L2 learner assimilates a target sound their L1 lacks to the nearest L1
category: a Bengali speaker says /dʒ/ero for /z/ero; a Hindi speaker merges
/v/~/w/; both map /θ/→/t̪/. These are not random errors — they are a directed,
per-L1, per-phoneme interference field.

We model each as an **inhibitory edge**  target ⟿ attractor  carrying a belief
P(this learner substitutes `attractor` for `target`). Unlike an ordinary CASCADE
edge (belief rises with evidence of the relation), a confusion edge is
**suppressed** as the learner masters the target contrast — "contrastive
percolation": acquiring /z/ actively drives down the /z/→/dʒ/ substitution
belief. The cause-net reads this to make `l1_interference` quantitative and to
name the exact substitution.

Grounding: bn/hi are calibrated from Indian-English contrastive phonology; ta/te
and the default set are literature-default (pan-Indian-English features), mirror-
ing the CIF attractor convention. All targets/attractors are REAL graph phoneme
node ids (see displays: phoneme-z=/z/, phoneme-j=/dʒ/, phoneme-th=/θ/, ...).
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from cognitive_twin import DB_PATH

# target_node → (attractor_node, population base-rate weight).  All ids exist in
# the graph. Weight = prior P(substitution) for a naive speaker of that L1.
CONFUSION: dict[str, dict[str, tuple[str, float]]] = {
    "bn": {   # Bengali
        "phoneme-z":  ("phoneme-j", 0.80),   # /z/ → /dʒ/  ("jero")
        "phoneme-v":  ("phoneme-b", 0.70),   # /v/ → /b/   (no /v/ in Bengali)
        "phoneme-th": ("phoneme-t", 0.65),   # /θ/ → /t̪/
        "phoneme-dh": ("phoneme-d", 0.65),   # /ð/ → /d̪/
        "phoneme-s":  ("phoneme-sh", 0.45),  # /s/ → /ʃ/   (Bengali sibilant bias)
    },
    "hi": {   # Hindi
        "phoneme-v":  ("phoneme-w", 0.75),   # /v/ ~ /w/ merger (ʋ)
        "phoneme-w":  ("phoneme-v", 0.75),   # symmetric
        "phoneme-z":  ("phoneme-j", 0.60),   # /z/ → /dʒ/  ("jyaada")
        "phoneme-th": ("phoneme-t", 0.65),   # /θ/ → /t̪/
        "phoneme-dh": ("phoneme-d", 0.65),   # /ð/ → /d̪/
        "phoneme-f":  ("phoneme-p", 0.50),   # /f/ → /pʰ/  ("phull")
    },
    "ta": {   # Tamil (Dravidian: voicing allophonic, few fricatives)
        "phoneme-z":  ("phoneme-j", 0.60),
        "phoneme-f":  ("phoneme-p", 0.55),
        "phoneme-sh": ("phoneme-s", 0.50),
        "phoneme-th": ("phoneme-t", 0.55),
        "phoneme-dh": ("phoneme-d", 0.55),
    },
    "te": {   # Telugu
        "phoneme-z":  ("phoneme-j", 0.55),
        "phoneme-f":  ("phoneme-p", 0.50),
        "phoneme-v":  ("phoneme-w", 0.55),
        "phoneme-th": ("phoneme-t", 0.55),
    },
}
# pan-Indian-English default for any other / unspecified L1
DEFAULT_CONFUSION = {
    "phoneme-th": ("phoneme-t", 0.55),
    "phoneme-dh": ("phoneme-d", 0.55),
    "phoneme-z":  ("phoneme-j", 0.50),
    "phoneme-v":  ("phoneme-w", 0.45),
}

RAISE = 0.30            # substitution evidence → confusion up
SUPPRESS = 0.40         # correct target production → confusion down
MASTERY_SUPPRESS = 0.50  # mastering the target contrast decays confusion
FLOOR, CEIL = 0.02, 0.98


def table_for(l1: str) -> dict[str, tuple[str, float]]:
    return CONFUSION.get(l1, DEFAULT_CONFUSION)


def attractor_for(l1: str, target_node: str) -> tuple[str, float] | None:
    """(attractor_node, prior_weight) if this L1 confuses `target_node`, else None."""
    return table_for(l1).get(target_node)


@dataclass
class ConfusionBelief:
    l1: str
    target: str
    attractor: str
    belief: float          # P(substitution)


def _connect() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.execute("""CREATE TABLE IF NOT EXISTS confusion (
        student_id TEXT NOT NULL,
        l1         TEXT NOT NULL,
        target     TEXT NOT NULL,
        attractor  TEXT NOT NULL,
        belief     REAL NOT NULL,
        updated_ts REAL NOT NULL,
        PRIMARY KEY (student_id, l1, target)
    )""")
    return c


def _read(c: sqlite3.Connection, student_id: str, l1: str,
          target_node: str) -> ConfusionBelief | None:
    entry = attractor_for(l1, target_node)
    if entry is None:
        return None
    attractor, prior = entry
    row = c.execute("SELECT attractor, belief FROM confusion "
                    "WHERE student_id=? AND l1=? AND target=?",
                    (student_id, l1, target_node)).fetchone()
    if row is None:
        return ConfusionBelief(l1, target_node, attractor, prior)
    return ConfusionBelief(l1, target_node, row[0], row[1])


def _upsert(c: sqlite3.Connection, student_id: str, b: ConfusionBelief) -> None:
    c.execute("INSERT INTO confusion VALUES (?,?,?,?,?,?) "
              "ON CONFLICT(student_id,l1,target) DO UPDATE SET "
              "belief=excluded.belief, attractor=excluded.attractor, "
              "updated_ts=excluded.updated_ts",
              (student_id, b.l1, b.target, b.attractor,
               max(FLOOR, min(CEIL, b.belief)), time.time()))


def get(student_id: str, l1: str, target_node: str) -> ConfusionBelief | None:
    """Current substitution belief; seeded from the population prior on first look."""
    c = _connect()
    try:
        return _read(c, student_id, l1, target_node)
    finally:
        c.close()


def note_production(student_id: str, l1: str, target_node: str,
                    correct: bool, confidence: float = 1.0) -> ConfusionBelief | None:
    """Evidence from a production attempt on the target phoneme.
    correct=False raises the substitution belief; correct=True suppresses it
    (contrastive suppression — the learner is acquiring the L1-absent contrast).
    Read-modify-write in ONE connection/transaction to stay consistent."""
    c = _connect()
    try:
        b = _read(c, student_id, l1, target_node)
        if b is None:
            return None
        if correct:
            b.belief = b.belief * (1 - confidence * SUPPRESS)
        else:
            b.belief = b.belief + (1 - b.belief) * confidence * RAISE
        b.belief = max(FLOOR, min(CEIL, b.belief))
        _upsert(c, student_id, b)
        c.commit()
        return b
    finally:
        c.close()


def suppress_on_mastery(student_id: str, l1: str, target_node: str,
                        node_mastery: float) -> None:
    """Contrastive percolation: mastering the target contrast decays confusion."""
    if node_mastery < 0.80:
        return
    c = _connect()
    try:
        b = _read(c, student_id, l1, target_node)
        if b is None:
            return
        b.belief = max(FLOOR, b.belief * (1 - MASTERY_SUPPRESS))
        _upsert(c, student_id, b)
        c.commit()
    finally:
        c.close()


def snapshot(student_id: str, l1: str) -> list[ConfusionBelief]:
    """All active confusion beliefs for this learner+L1, strongest first."""
    out = []
    for target in table_for(l1):
        b = get(student_id, l1, target)
        if b:
            out.append(b)
    out.sort(key=lambda x: -x.belief)
    return out
