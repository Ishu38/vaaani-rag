"""Does the contrastive confusion field make l1_interference DISCRIMINATIVE?

Claim: with confusion edges, the cause-net attributes l1_interference much more
strongly to a miss on a phoneme the learner's L1 lacks (e.g. Bengali /z/) than
to a miss on a phoneme the L1 shares (e.g. /m/) — a per-phoneme, quantitative
signal the old binary script-based L1 heuristic could not give.

We simulate learners of an L1, drive a miss on each confused and each shared
phoneme, run the REAL cause_net.diagnose, and measure how well the resulting
l1_interference posterior separates confused-from-shared phoneme misses (AUC).
Real modules, real DB (throwaway ids, cleaned). Seeded.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import l1_confusion as lc
import cause_net
from development_engine import WorldModel
from evidence_graph import EvidenceObject
import cognitive_twin as twin
from cognitive_twin import DB_PATH

PREFIX = "__cc_"


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def clean():
    c = sqlite3.connect(DB_PATH)
    for t in ("twin", "twin_edge", "predictions", "evidence", "diagnoses", "confusion"):
        try:
            c.execute(f"DELETE FROM {t} WHERE student_id LIKE ?", (PREFIX + "%",))
        except Exception:
            pass
    c.commit(); c.close()


def run(l1="bn", n_learners=60, seed=3):
    world = WorldModel()
    confused = [n for n in lc.table_for(l1) if n in world.nodes]
    shared = [n for n in world.nodes
              if n.startswith("phoneme-") and n not in lc.table_for(l1)][:len(confused) + 4]
    scores, labels, byname = [], [], {}
    for i in range(n_learners):
        for group, is_conf in ((confused, 1), (shared, 0)):
            for ph in group:
                sid = f"{PREFIX}{l1}_{i}_{ph}"
                # a couple of production misses on this phoneme (raises confusion iff confused)
                for _ in range(2):
                    twin.update(EvidenceObject(sid, ph, source="mission",
                                               outcome="incorrect", confidence=0.9,
                                               meta={"l1": l1}))
                    if is_conf:
                        lc.note_production(sid, l1, ph, correct=False, confidence=0.9)
                d = cause_net.diagnose(sid, ph, world, outcome="incorrect", l1=l1, persist=False)
                p = d.distribution["l1_interference"]
                scores.append(p); labels.append(is_conf)
                byname.setdefault(is_conf, []).append(p)
    clean()
    mean_conf = sum(byname[1]) / len(byname[1])
    mean_shared = sum(byname[0]) / len(byname[0])
    return {
        "l1": l1,
        "n_confused_phonemes": len(confused),
        "n_shared_phonemes": len(shared),
        "mean_l1_posterior_confused": round(mean_conf, 4),
        "mean_l1_posterior_shared": round(mean_shared, 4),
        "separation": round(mean_conf - mean_shared, 4),
        "auc_confused_vs_shared": round(auc(scores, labels), 4),
    }


def main():
    clean()
    import json
    out = {}
    for l1 in ("bn", "hi"):
        r = run(l1)
        out[l1] = r
        print(f"\nL1={r['l1']}  ({r['n_confused_phonemes']} confused vs "
              f"{r['n_shared_phonemes']} shared phonemes)")
        print(f"  mean l1_interference posterior — confused: "
              f"{r['mean_l1_posterior_confused']:.3f}  shared: {r['mean_l1_posterior_shared']:.3f}")
        print(f"  separation: {r['separation']:+.3f}   AUC(confused vs shared): "
              f"{r['auc_confused_vs_shared']:.3f}")
    p = Path(__file__).resolve().parent / "contrastive_confusion_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
