"""Pronunciation ear → evidence bridge (perception channel for V_Φ).

Position (LINGUISTIC_STATE_THEORY.md §2): this is 𝒫 for the phonology tier.
The CAVP-lite engine (`/api/pronounce/check`, prompted-speech lane: quality
gate → wav2vec2 forced alignment vs the KNOWN target → IPA decode → NW
pairing) returns per-phone outcomes with a confidence derived from measured
audio quality. That confidence flows straight into Jeffrey conditioning
(twin + credal), so a noisy recording mathematically counts for less than a
clean one — the theory's showcase of the w parameter.

Abstention duty (law A2 / operator 𝒫): engine statuses "unmeasurable" and
"error" emit NOTHING. Per-phone evidence attaches only to phoneme nodes that
exist in the graph (e.g. id "phoneme-f" or display "f"); unmapped phones are
skipped and counted, never guessed.

Engine placement is deployment config: VAAANI_EAR_URL + VAAANI_EAR_KEY.
Unset ⇒ the endpoint reports the channel unavailable (503); nothing else in
the loop depends on it.
"""

from __future__ import annotations

import os
import re

import httpx

EAR_URL = os.environ.get("VAAANI_EAR_URL", "")          # e.g. http://127.0.0.1:8005
EAR_KEY = os.environ.get("VAAANI_EAR_KEY", "")
EAR_TIMEOUT = float(os.environ.get("VAAANI_EAR_TIMEOUT", "60"))
MAX_AUDIO_BYTES = 4 * 1024 * 1024                        # ~2 s clips; hard cap

# v0 word-level outcome thresholds on the engine's match_ratio.
MATCH_CORRECT = 0.80
MATCH_PARTIAL = 0.50


class EarUnavailable(Exception):
    pass


def check_with_engine(audio_bytes: bytes, filename: str, target_text: str,
                      language: str = "en") -> dict:
    """One round-trip to the CAVP-lite engine. Raises EarUnavailable when the
    channel is unconfigured or unreachable — the caller maps that to 503."""
    if not EAR_URL:
        raise EarUnavailable("VAAANI_EAR_URL not configured")
    try:
        with httpx.Client(timeout=EAR_TIMEOUT) as cl:
            r = cl.post(
                EAR_URL.rstrip("/") + "/api/pronounce/check",
                headers={"X-Engine-API-Key": EAR_KEY} if EAR_KEY else {},
                files={"audio": (filename or "clip.webm", audio_bytes)},
                data={"target_text": target_text, "language": language},
            )
    except httpx.HTTPError as e:
        raise EarUnavailable(f"ear engine unreachable: {e}") from e
    if r.status_code != 200:
        raise EarUnavailable(f"ear engine HTTP {r.status_code}")
    return r.json()


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.lower())


def phone_node_index(world) -> dict[str, str]:
    """phone token → phoneme node id, built once per WorldModel instance.
    Matches node id 'phoneme-<phone>' or a node whose display equals the
    phone token exactly (IPA symbols are their own displays in the graph)."""
    idx = getattr(world, "_ear_phone_index", None)
    if idx is None:
        idx = {}
        for nid, node in world.nodes.items():
            if nid.startswith("phoneme-"):
                idx.setdefault(_norm(nid[len("phoneme-"):]), nid)
            if node.get("type") == "phoneme":
                disp = node.get("display", "")
                if disp:
                    idx.setdefault(_norm(disp), nid)
        world._ear_phone_index = idx
    return idx


def ingest_pronunciation(student_id: str, node_id: str, result: dict,
                         world) -> dict:
    """Convert one engine result into evidence and fold it into the twin
    (BKT + credal share the single ingress). Returns a summary the SPA can
    show; {"recorded": False} means honest abstention, not failure."""
    status = result.get("status")
    if status != "ok":
        return {"recorded": False,
                "reason": result.get("reason") or result.get("error")
                or f"engine status {status!r}"}

    import cognitive_twin as twin
    from evidence_graph import EvidenceObject

    conf = float(result.get("confidence", 0.0))
    match_ratio = float(result.get("match_ratio", 0.0))
    target_text = result.get("target_text", "")

    # Word-level score = fraction of target phones produced acceptably, from
    # the same per-phone outcomes the phone evidence uses. The engine's
    # match_ratio deducts insertion/deletion events that consume no target
    # phone, so it collapses to 0 on noisy decodes even when most target
    # phones landed (measured on-VM 2026-07-12: 5/7 correct, match_ratio 0.0).
    phone_ev = result.get("evidence", [])
    if phone_ev:
        word_score = sum(1 for e in phone_ev if e.get("outcome") == "correct") / len(phone_ev)
    else:
        word_score = match_ratio
    if word_score >= MATCH_CORRECT:
        outcome = "correct"
    elif word_score >= MATCH_PARTIAL:
        outcome = "partial"
    else:
        outcome = "incorrect"
    word_belief = twin.update(EvidenceObject(
        student_id=student_id, node_id=node_id, source="audio",
        outcome=outcome, confidence=conf,
        meta={"target_text": target_text, "match_ratio": match_ratio,
              "word_score": round(word_score, 3), "level": "word",
              "events_by_kind": result.get("events_by_kind"),
              "low_confidence": result.get("low_confidence")},
    ))

    # Phone-level evidence on phoneme nodes that exist in the graph.
    idx = phone_node_index(world)
    phone_updates = []
    skipped = 0
    for item in result.get("evidence", []):
        phone = item.get("phone", "")
        pnode = idx.get(_norm(phone)) if phone else None
        if pnode is None:
            skipped += 1
            continue
        b = twin.update(EvidenceObject(
            student_id=student_id, node_id=pnode, source="audio",
            outcome=item.get("outcome", "incorrect"),
            confidence=float(item.get("confidence", conf)),
            meta={"target_text": target_text, "phone": phone, "level": "phone"},
        ))
        phone_updates.append({"phone": phone, "node_id": pnode,
                              "outcome": item.get("outcome"),
                              "mastery": round(b.mastery, 4)})

    return {"recorded": True,
            "outcome": outcome,
            "word_score": round(word_score, 3),
            "match_ratio": match_ratio,
            "confidence": conf,
            "low_confidence": bool(result.get("low_confidence")),
            "word_update": {"node_id": word_belief.node_id,
                            "mastery": round(word_belief.mastery, 4),
                            "exposures": word_belief.exposures},
            "phone_updates": phone_updates,
            "phones_unmapped": skipped}
