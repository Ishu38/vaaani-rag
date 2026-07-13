"""Demo: one full turn of the cognitive loop on the REAL language graph.

    Evidence -> Cognitive Twin -> Development Engine (frontier) ->
    Pedagogical Planner (mission + logged prediction) ->
    [learner attempts] -> Evidence -> Twin update -> Metacognitive calibration

Run:  cd backend && python demo_cognitive_loop.py
"""

from __future__ import annotations

import random

from evidence_graph import EvidenceObject
import cognitive_twin as twin
from development_engine import WorldModel, frontier
from pedagogical_planner import select_activity

STUDENT = "demo_student_riya"
rng = random.Random(7)


def main() -> None:
    world = WorldModel()
    print(f"Linguistic World Model: {len(world.nodes)} nodes, "
          f"{sum(len(v) for v in world.prereqs.values())} depends_on edges\n")

    # ── Day 1: seed evidence — Riya answers a placement quiz ──
    seed_nodes = [n for n, pre in world.prereqs.items() if pre][:1]
    # pick a node WITH prerequisites and seed its prereqs as known
    target = seed_nodes[0]
    prereqs = world.prereqs[target]
    print(f"Target node: '{world.display(target)}'  prerequisites: "
          f"{[world.display(p) for p in prereqs]}")

    for p in prereqs:
        for _ in range(4):  # four correct answers on each prerequisite
            twin.update(EvidenceObject(STUDENT, p, "quiz", "correct"))
    print("\nAfter placement quiz (4 correct per prerequisite):")
    for p in prereqs:
        print(f"  mastery[{world.display(p)}] = {twin.get(STUDENT, p).mastery:.3f}")
    print(f"  mastery[{world.display(target)}] = {twin.get(STUDENT, target).mastery:.3f} (untouched prior)")

    # ── Development Engine: where is Riya's frontier now? ──
    fr = frontier(STUDENT, world, limit=5)
    print(f"\nZPD frontier (top {len(fr)}):")
    for f in fr:
        print(f"  {f.display:32s} P(success)={f.p_success:.2f} "
              f"readiness={f.readiness:.2f}")

    # ── Pedagogical Planner: one mission, with its reason ──
    mission = select_activity(STUDENT, world)
    print(f"\nMission decision: [{mission.kind}] '{mission.display}'")
    print(f"  reason: {mission.reason}")
    print(f"  prediction logged: P(success)={mission.p_success:.2f}")

    # ── Learner Interaction Loop: Riya attempts 6 missions over time ──
    print("\nSimulating 6 attempts on the mission node "
          "(true skill grows as she practices):")
    for i in range(6):
        p_true = min(0.9, 0.35 + 0.12 * i)      # her real ability improves
        outcome = "correct" if rng.random() < p_true else "incorrect"
        b = twin.update(EvidenceObject(STUDENT, mission.node_id, "mission", outcome))
        print(f"  attempt {i+1}: {outcome:9s} -> mastery {b.mastery:.3f}")
        nxt = select_activity(STUDENT, world)
        if nxt and nxt.node_id != mission.node_id:
            print(f"    planner moved on: [{nxt.kind}] '{nxt.display}' — {nxt.reason[:60]}…")

    # ── Metacognitive Evaluation: was the system's confidence honest? ──
    cal = twin.calibration(STUDENT)
    print("\nCalibration (predicted P(success) vs actual rate):")
    for row in cal:
        print(f"  {row['bin']}: n={row['n']}, predicted {row['predicted_mean']:.2f}, "
              f"actual {row['actual_rate']:.2f}")

    print(f"\nEvidence objects recorded: "
          f"{__import__('evidence_graph').count(STUDENT)}")
    print("Loop closed: evidence → twin → frontier → mission → evidence. ✔")


if __name__ == "__main__":
    main()
