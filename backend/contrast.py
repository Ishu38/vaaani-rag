"""Contrast engine — WHY a sound is meaningful, and WHY a swap is wrong.

Neil's requirement: a child must build the INNATE understanding of why a sound
matters and why theirs is wrong — not just "correct/incorrect". Phonology gives
two first-principles answers:

  • WHY meaningful — the MINIMAL PAIR. A phoneme earns its status because
    swapping it makes a *different word*: sip → ship, vine → wine, thin → tin.
    That functional load is the reason the contrast exists at all.

  • WHY wrong — the ARTICULATORY FEATURE that differs. Two sounds differ in
    voicing, place, or manner. The child can FEEL that difference in the Sound
    Lab (turn the voice on, move the tongue, change the airflow). Computed from
    the real graph attributes {voice, place, manner} on each phoneme node.

No LLM. Data-driven from the graph + a curated minimal-pair table.
"""

from __future__ import annotations

# Minimal pairs keyed by the UNORDERED phoneme-node pair → the word that carries
# each phoneme (same slot, only that sound differs → a different word).
MINIMAL_PAIRS: dict[frozenset, dict[str, str]] = {
    frozenset({"phoneme-s", "phoneme-sh"}): {"phoneme-s": "sip",  "phoneme-sh": "ship"},
    frozenset({"phoneme-s", "phoneme-z"}):  {"phoneme-s": "sip",  "phoneme-z": "zip"},
    frozenset({"phoneme-z", "phoneme-j"}):  {"phoneme-z": "zest", "phoneme-j": "jest"},
    frozenset({"phoneme-v", "phoneme-w"}):  {"phoneme-v": "vine", "phoneme-w": "wine"},
    frozenset({"phoneme-f", "phoneme-v"}):  {"phoneme-f": "fan",  "phoneme-v": "van"},
    frozenset({"phoneme-f", "phoneme-p"}):  {"phoneme-f": "fan",  "phoneme-p": "pan"},
    frozenset({"phoneme-th", "phoneme-t"}): {"phoneme-th": "thin", "phoneme-t": "tin"},
    frozenset({"phoneme-th", "phoneme-s"}): {"phoneme-th": "thin", "phoneme-s": "sin"},
    frozenset({"phoneme-dh", "phoneme-d"}): {"phoneme-dh": "they", "phoneme-d": "day"},
    frozenset({"phoneme-dh", "phoneme-z"}): {"phoneme-dh": "then", "phoneme-z": "zen"},
    frozenset({"phoneme-t", "phoneme-d"}):  {"phoneme-t": "ten",  "phoneme-d": "den"},
    frozenset({"phoneme-p", "phoneme-b"}):  {"phoneme-p": "pat",  "phoneme-b": "bat"},
    frozenset({"phoneme-k", "phoneme-g"}):  {"phoneme-k": "coat", "phoneme-g": "goat"},
    frozenset({"phoneme-sh", "phoneme-ch"}): {"phoneme-sh": "ship", "phoneme-ch": "chip"},
    frozenset({"phoneme-j", "phoneme-ch"}): {"phoneme-j": "jeep", "phoneme-ch": "cheap"},
    frozenset({"phoneme-l", "phoneme-r"}):  {"phoneme-l": "light", "phoneme-r": "right"},
    frozenset({"phoneme-n", "phoneme-ng"}): {"phoneme-n": "sin",  "phoneme-ng": "sing"},
    frozenset({"phoneme-w", "phoneme-y"}):  {"phoneme-w": "wet",  "phoneme-y": "yet"},
    frozenset({"phoneme-b", "phoneme-v"}):  {"phoneme-b": "bat",  "phoneme-v": "vat"},
}

# How each feature feels — the first-principles instruction the child can act on.
_FEEL = {
    "voicing": "turn your voice on or off — touch your throat and feel it buzz for one, stay silent for the other",
    "place":   "move where your tongue or lips make the sound",
    "manner":  "change how the air moves — stop it completely, let it hiss, or let it flow",
}


def _attr(world, node_id: str) -> dict:
    n = world.nodes.get(node_id, {})
    return {"voice": n.get("voice"), "place": n.get("place"),
            "manner": n.get("manner"), "ipa": n.get("display", node_id)}


def feature_difference(world, node_a: str, node_b: str) -> list[dict]:
    """The articulatory features that differ between two phonemes, from the graph."""
    a, b = _attr(world, node_a), _attr(world, node_b)
    diffs = []
    if a["voice"] is not None and b["voice"] is not None and a["voice"] != b["voice"]:
        diffs.append({"feature": "voicing",
                      "a": "voiced (buzzing)" if a["voice"] else "voiceless (whispered)",
                      "b": "voiced (buzzing)" if b["voice"] else "voiceless (whispered)",
                      "feel": _FEEL["voicing"]})
    if a["place"] and b["place"] and a["place"] != b["place"]:
        diffs.append({"feature": "place", "a": a["place"], "b": b["place"],
                      "feel": _FEEL["place"]})
    if a["manner"] and b["manner"] and a["manner"] != b["manner"]:
        diffs.append({"feature": "manner", "a": a["manner"], "b": b["manner"],
                      "feel": _FEEL["manner"]})
    return diffs


def contrast(world, ref_node: str, got_node: str) -> dict | None:
    """WHY the target sound is meaningful and WHY the learner's swap is wrong.

    ref_node = the sound the word needs; got_node = the sound the learner made.
    Returns None when we can't say anything grounded (no data / same sound)."""
    if not ref_node or not got_node or ref_node == got_node:
        return None
    ref = _attr(world, ref_node)
    got = _attr(world, got_node)
    diffs = feature_difference(world, ref_node, got_node)

    mp = MINIMAL_PAIRS.get(frozenset({ref_node, got_node}))
    why_meaningful = None
    if mp:
        wa, wb = mp.get(ref_node), mp.get(got_node)
        why_meaningful = (
            f"“{wa}” and “{wb}” are different words — the only thing that changes "
            f"one into the other is this single sound. That is why {ref['ipa']} matters.")

    if diffs:
        parts = []
        for d in diffs:
            if d["feature"] == "voicing":
                parts.append(f"{ref['ipa']} is {d['a']} but you made it {got['ipa']}, which is {d['b']}")
            else:
                parts.append(f"the {d['feature']} moves from {d['a']} ({ref['ipa']}) to {d['b']} ({got['ipa']})")
        why_wrong = "Here's the difference: " + "; ".join(parts) + "."
        feel = diffs[0]["feel"]
    else:
        why_wrong = f"You made {got['ipa']} where the word needs {ref['ipa']}."
        feel = None

    return {
        "ref": ref_node, "got": got_node,
        "ref_ipa": ref["ipa"], "got_ipa": got["ipa"],
        "minimal_pair": mp,
        "feature_difference": diffs,
        "why_meaningful": why_meaningful,
        "why_wrong": why_wrong,
        "feel_it": feel,
    }
