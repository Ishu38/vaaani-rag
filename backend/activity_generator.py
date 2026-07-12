"""Procedural Activity Generator — MissionDecision -> learner-facing activity.

Architecture position (Neil's diagram): box 7. **Contains no LLM and no SLM.**
Every activity is generated symbolically from the Linguistic World Model:

  - morphology tier (Kurdi ch. 3): word-building from `root_of` /
    `word_family` edges — the FST-lattice idea in template form
  - speech tier (Kurdi ch. 2): minimal-pair discrimination from
    `sounds_like` edges
  - meaning/bridge tier: `translates_to` / `means` edges (home-language
    bridge — Hindi/Bangla)
  - discovery fallback: neighborhood exploration mission over any edges

Deterministic: same twin state + same graph -> same activity. Every activity
declares the evidence it will emit (`credits` = node ids), so the Learner
Interaction Loop knows exactly what to record on completion — the loop stays
closed by construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from pedagogical_planner import MissionDecision
from development_engine import WorldModel

try:
    from config import GRAPH_PATH
except ImportError:
    GRAPH_PATH = Path(__file__).resolve().parent.parent / "data" / "graph.json"


@dataclass
class Activity:
    activity_type: str            # word_building | minimal_pair | bridge | discovery | probe | review_recall
    node_id: str
    title: str
    instructions: str             # plain language, learner-facing
    items: list = field(default_factory=list)
    credits: list = field(default_factory=list)   # node ids to emit evidence for
    reason: str = ""              # planner's explanation, surfaced to teachers

    def to_dict(self) -> dict:
        return asdict(self)


class _Edges:
    """Typed neighborhood lookup over the raw graph."""

    def __init__(self, path=GRAPH_PATH):
        g = json.loads(Path(path).read_text())
        self.by_type: dict[str, list[tuple[str, str]]] = {}
        for e in g["links"]:
            self.by_type.setdefault(e.get("type", "?"), []).append((e["source"], e["target"]))

    def neighbors(self, node_id: str, edge_type: str) -> list[str]:
        out = []
        for s, t in self.by_type.get(edge_type, []):
            if s == node_id:
                out.append(t)
            elif t == node_id:
                out.append(s)
        return out


_EDGES: _Edges | None = None


def get_edges() -> _Edges:
    """Process-wide edge cache (the graph is read-only at runtime)."""
    global _EDGES
    if _EDGES is None:
        _EDGES = _Edges()
    return _EDGES


def node_tier(node_id: str, edges: _Edges | None = None) -> str:
    """Which linguistic tier would this node's activity live in?
    Mirrors generate()'s dispatch so the planner can rotate tiers."""
    edges = edges or get_edges()
    if edges.neighbors(node_id, "root_of") or len(edges.neighbors(node_id, "word_family")) >= 2:
        return "morph"
    if edges.neighbors(node_id, "sounds_like"):
        return "sound"
    if edges.neighbors(node_id, "translates_to") or edges.neighbors(node_id, "means"):
        return "bridge"
    return "other"


def generate(mission: MissionDecision, world: WorldModel | None = None,
             edges: _Edges | None = None) -> Activity:
    """Turn a planner decision into a concrete activity. Symbolic only."""
    world = world or WorldModel()
    edges = edges or get_edges()
    nid, disp = mission.node_id, mission.display

    if mission.kind == "probe":
        return Activity(
            "probe", nid, f"Quick check: {disp}",
            f"Try this one — it helps me understand what you already know about “{disp}”.",
            items=[{"prompt": f"Show or tell what you know about “{disp}”."}],
            credits=[nid], reason=mission.reason)

    if mission.kind == "review":
        return Activity(
            "review_recall", nid, f"Remember {disp}?",
            f"We met “{disp}” before. Let's see if it stuck!",
            items=[{"prompt": f"Use “{disp}” in one example of your own."}],
            credits=[nid], reason=mission.reason)

    # ── learn missions: pick the richest tier the graph supports.
    # Verifiable-by-construction where the graph holds an answer key:
    # bridge_match and sound_detective carry the correct answer (from edges)
    # plus deterministic distractors — the UI checks the tap, and the twin
    # gets measured evidence instead of an honor-system button.
    roots = edges.neighbors(nid, "root_of")
    family = edges.neighbors(nid, "word_family")
    sounds = edges.neighbors(nid, "sounds_like")
    bridges = edges.neighbors(nid, "translates_to") + edges.neighbors(nid, "means")

    def _det_shuffle(opts: list, seed: str) -> list:
        return sorted(opts, key=lambda o: hash((seed, o)) % 9973)

    if roots or len(family) >= 2:
        members = (roots + family)[:5]
        return Activity(
            "word_building", nid, f"Word detectives: {disp}",
            f"These words all share a piece with “{disp}”. Find the shared piece, "
            f"then build one more word of your own that uses it.",
            items=[{"word": world.display(m)} for m in members],
            credits=[nid] + members[:2], reason=mission.reason)

    if sounds:
        # never let the answer (or any option) echo the anchor itself
        sound_names = [s for s in sounds
                       if world.display(s).lower() != disp.lower()]
        if not sound_names:
            sound_names = sounds
        correct = world.display(sound_names[0])
        pool = [world.display(n) for t in ("translates_to", "used_in", "is_a")
                for s_, t_ in edges.by_type.get(t, [])[:40] for n in (s_, t_)
                if n in world.nodes and n not in sounds and n != nid]
        distractors = [d for d in dict.fromkeys(pool) if d.lower() != correct.lower()][:2]
        options = _det_shuffle([correct] + distractors, nid)
        return Activity(
            "sound_detective", nid, f"Sound detective: {disp}",
            f"Say “{disp}” out loud. Now say each word below out loud too. "
            f"ONE of them shares its sound with “{disp}” — trust your ears, then tap it.",
            items=[{"prompt": f"Which one sounds like “{disp}”?",
                    "options": options, "answer": correct}],
            credits=[nid, sounds[0]], reason=mission.reason)

    if bridges:
        # Prefer a true home-language target (Indic script) over Latin-script
        # 'means' relations — the bridge must actually cross languages.
        def _is_indic(node: str) -> bool:
            return any(ord(ch) > 0x0900 for ch in world.display(node))
        indic = [b for b in bridges if _is_indic(b)]
        correct = world.display((indic or bridges)[0])
        correct_is_indic = any(ord(ch) > 0x0900 for ch in correct)
        other_bridges = [world.display(t) for s, t in
                         edges.by_type.get("translates_to", [])
                         if s != nid and t != nid and t in world.nodes]
        if correct_is_indic:
            # distractors must be same-script translations of OTHER words,
            # or the correct answer gives itself away by script alone
            other_bridges = [d for d in other_bridges
                             if any(ord(ch) > 0x0900 for ch in d)]
        distractors = [d for d in dict.fromkeys(other_bridges)
                       if d.lower() != correct.lower()][:2]
        options = _det_shuffle([correct] + distractors, nid)
        return Activity(
            "bridge_match", nid, f"Home-language bridge: {disp}",
            f"“{disp}” has a friend in your home language. "
            f"Which of these is it? Say each one aloud before you choose.",
            items=[{"prompt": f"Find the home-language friend of “{disp}”",
                    "options": options, "answer": correct}],
            credits=[nid], reason=mission.reason)

    neighborhood = []
    for et in ("used_in", "is_a", "written_as", "cognate_with"):
        neighborhood += edges.neighbors(nid, et)
    return Activity(
        "discovery", nid, f"Explore: {disp}",
        f"Follow “{disp}” through the word web. Find two connections and "
        f"explain each in one sentence.",
        items=[{"connection": world.display(n)} for n in neighborhood[:4]] or
              [{"connection": "search the word web"}],
        credits=[nid], reason=mission.reason)
