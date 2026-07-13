"""L1 Graft — CASCADE home-language graph isomorphism.

Maps the child's L1 (Hindi, Bengali, Tamil etc.) onto the target language
graph via `translates_to` and `cognate_with` edges. An attested mapping
bootstraps the edge belief: the child already has an isomorphic subgraph.

A Bengali-speaking child learning English gets a head start on every
edge where the graph shows an attested cognate or translation — CASCADE
quantifies exactly how much head start, with a confidence score per mapping.
"""

from __future__ import annotations

# Unicode script ranges for common Indian languages
L1_SCRIPTS = {
    "hi":  (0x0900, 0x097F),   # Devanagari (Hindi, Marathi, Sanskrit)
    "bn":  (0x0980, 0x09FF),   # Bengali
    "ta":  (0x0B80, 0x0BFF),   # Tamil
    "te":  (0x0C00, 0x0C7F),   # Telugu
    "gu":  (0x0A80, 0x0AFF),   # Gujarati
    "pa":  (0x0A00, 0x0A7F),   # Gurmukhi (Punjabi)
    "ml":  (0x0D00, 0x0D7F),   # Malayalam
    "kn":  (0x0C80, 0x0CFF),   # Kannada
    "or":  (0x0B00, 0x0B7F),   # Odia
}

# L1 language codes to display names
L1_NAMES = {
    "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "gu": "Gujarati", "pa": "Punjabi", "ml": "Malayalam",
    "kn": "Kannada", "or": "Odia", "en": "English",
}

# Confidence tiers for mapping attestation
# High: edge exists in curriculum + has cross-linguistic validation
# Medium: edge exists in curriculum only
# Low: inferred from shared script/semantics
MAPPING_CONFIDENCE = {
    "translates_to": 0.85,   # Explicitly curated translation pairs
    "cognate_with": 0.70,    # Sanskrit-derived cognates
}


def _is_l1_script(node_display: str, l1: str) -> bool:
    """Check if a node's display text contains characters from the L1 script."""
    if l1 not in L1_SCRIPTS:
        return False
    lo, hi = L1_SCRIPTS[l1]
    return any(lo <= ord(ch) <= hi for ch in node_display)


def compute_l1_graft(
    graph_edges: dict,
    node_display: callable,
    l1: str,
    min_confidence: float = 0.5,
) -> dict[str, float]:
    """Compute initial α boost for edges with L1 graft pre-images.

    Args:
        graph_edges: {etype: [(source, target), ...]}
        node_display: function(id) -> display_name
        l1: language code ('hi', 'bn', etc.)
        min_confidence: filter threshold

    Returns:
        {edge_key: alpha_boost} — the boost to add to the initial α
        when seeding the twin for a new learner with this L1.
    """
    if l1 == "en":
        return {}

    boosts: dict[str, float] = {}

    for etype in ("translates_to", "cognate_with"):
        base_confidence = MAPPING_CONFIDENCE.get(etype, 0.5)
        if base_confidence < min_confidence:
            continue

        pairs = graph_edges.get(etype, [])
        for source, target in pairs:
            tgt_display = node_display(target)
            src_display = node_display(source)

            # The L1 node is the one with Indic script characters
            if _is_l1_script(tgt_display, l1):
                confidence = base_confidence
                if etype == "cognate_with":
                    confidence += 0.05
                elif etype == "translates_to" and _is_l1_script(src_display, l1):
                    # Both sides are L1 → lower confidence for indirect mapping
                    confidence -= 0.10

                if confidence >= min_confidence:
                    key = f"{source}::{target}::{etype}"
                    boosts[key] = round(confidence * 2.0, 2)  # scale to α units

    return boosts


def l1_boost_factor(l1: str) -> float:
    """Global head-start factor based on L1.

    Returns a multiplier on the initial seeded α for edges.
    Languages with more attested cognates get higher boosts.
    """
    factors = {
        "hi": 1.5,   # Hindi — substantial Sanskrit cognate overlap
        "bn": 1.5,   # Bengali — same Indo-Aryan family
        "gu": 1.4,   # Gujarati
        "pa": 1.4,   # Punjabi
        "or": 1.3,   # Odia
        "ta": 1.1,   # Tamil — Dravidian, fewer cognates but translations exist
        "te": 1.1,   # Telugu
        "ml": 1.1,   # Malayalam
        "kn": 1.1,   # Kannada
        "en": 1.0,   # English L1 — no boost (native speaker)
    }
    return factors.get(l1, 1.0)


def seed_edges_from_l1(
    student_id: str,
    graph_edges: dict,
    node_display: callable,
    l1: str,
) -> int:
    """Seed edge-level twin beliefs from L1 graft on first encounter.

    Returns the number of edges seeded.
    """
    from evidence_graph import EvidenceObject
    import cognitive_twin as twin

    boosts = compute_l1_graft(graph_edges, node_display, l1)
    count = 0

    for edge_key, alpha_boost in boosts.items():
        src, tgt, et = twin._parse_edge_key(edge_key)

        # Seed node beliefs for both endpoints (CASCADE: L1 words are known)
        for nid in (src, tgt):
            nb = twin.get(student_id, nid)
            if nb.exposures == 0:
                ev = EvidenceObject(
                    student_id, nid, "mission", "correct",
                    confidence=0.90,
                    meta={"seed": True, "reason": "L1_graft", "l1": l1},
                )
                twin.update(ev)

        bel = twin.get_edge(student_id, edge_key)
        if bel.exposures > 0:
            continue

        import time
        now = time.time()
        from cognitive_twin import _conn
        seeded_mastery = twin.PRIOR + (1 - twin.PRIOR) * alpha_boost * 0.5
        seeded_mastery = min(0.80, seeded_mastery)

        with _conn() as c:
            c.execute("INSERT OR REPLACE INTO twin_edge VALUES (?,?,?,?,?,?,?,?) ",
                      (student_id, edge_key, src, tgt, et,
                       seeded_mastery, 0, now))
        count += 1

    return count
