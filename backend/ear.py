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


def ingest_say(student_id: str, word: str, result: dict, world,
               l1: str = "en") -> dict:
    """Voice-first door — the child SAYS an arbitrary word; the ear measures which
    sounds they produced. Returns per-sound feedback in the child's own letters
    (akshara) with the first-principles 'why' for any sound that drifted, folding
    the outcomes into the twin + confusion field. No typing, no symbols required —
    the accessible entry for a vernacular-medium learner."""
    status = result.get("status")
    if status != "ok":
        return {"recorded": False,
                "reason": result.get("reason") or result.get("error")
                or f"engine status {status!r}"}

    import cognitive_twin as twin
    from evidence_graph import EvidenceObject
    idx = phone_node_index(world)

    per_sound, mapped_ok, mapped_total = [], 0, 0
    for item in result.get("evidence", []):
        phone = item.get("phone", "")
        outcome = item.get("outcome", "incorrect")
        conf = float(item.get("confidence", 0.6))
        pnode = idx.get(_norm(phone)) if phone else None
        if pnode is None:
            per_sound.append({"ipa": phone, "node": None, "outcome": outcome,
                              "akshara": None, "status": "other"})
            continue
        correct = outcome == "correct"
        mapped_total += 1
        mapped_ok += 1 if correct else 0
        b = twin.update(EvidenceObject(
            student_id=student_id, node_id=pnode, source="audio",
            outcome="correct" if correct else "incorrect", confidence=conf,
            meta={"word": word, "phone": phone, "level": "phone", "channel": "say"}))
        entry = {"ipa": world.display(pnode), "node": pnode, "outcome": outcome,
                 "mastery": round(b.mastery, 4)}
        if l1 and l1 != "en":
            try:
                import l1_confusion as lc
                lc.note_production(student_id, l1, pnode, correct=correct, confidence=conf)
                lc.suppress_on_mastery(student_id, l1, pnode, b.mastery)
                aksh, st = (None, None)
                try:
                    import l1_script
                    if l1_script.supported(l1):
                        aksh, st = l1_script.akshara_for_consonant(l1, pnode)
                except Exception:
                    pass
                entry["akshara"], entry["status"] = aksh, st
                # inferred contrast: a missed sound their L1 lacks → why it's new
                if not correct:
                    att = lc.attractor_for(l1, pnode)
                    if att:
                        import contrast
                        entry["contrast"] = contrast.contrast(world, pnode, att[0])
            except Exception:
                pass
        per_sound.append(entry)

    return {"recorded": True, "word": word,
            "matched": mapped_ok, "total": mapped_total,
            "score": round(mapped_ok / mapped_total, 3) if mapped_total else 0.0,
            "low_confidence": bool(result.get("low_confidence")),
            "per_sound": per_sound}


def _word_phone_edges(word_id: str, pnode: str, graph_edges: dict) -> list[str]:
    """CASCADE: the graph edges that literally connect this word to this
    phoneme (either direction, any type) — e.g. phone —sounds_like→ f, or
    f —written_as→ ph. Pronouncing the word is direct evidence about these
    relationships, so the phone outcome updates their edge beliefs."""
    out: list[str] = []
    for etype, pairs in graph_edges.items():
        for s, t in pairs:
            if (s == word_id and t == pnode) or (s == pnode and t == word_id):
                out.append(f"{s}::{t}::{etype}")
    return out


def ingest_pronunciation(student_id: str, node_id: str, result: dict,
                         world, l1: str = "en") -> dict:
    """Convert one engine result into evidence and fold it into the twin
    (BKT + credal share the single ingress). Returns a summary the SPA can
    show; {"recorded": False} means honest abstention, not failure.

    CASCADE: per-phone outcomes also update the word↔phoneme *edge* beliefs,
    so a mispronunciation weakens the exact sound relationship (and CASCADE
    propagation carries it to neighbouring sound edges). On a wrong/partial
    word, the causal net diagnoses *why* from those same edge beliefs."""
    status = result.get("status")
    if status != "ok":
        return {"recorded": False,
                "reason": result.get("reason") or result.get("error")
                or f"engine status {status!r}"}

    import cognitive_twin as twin
    from development_engine import get_world_edges
    from evidence_graph import EvidenceObject
    graph_edges = get_world_edges()

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
    edge_updates = []
    skipped = 0
    for item in result.get("evidence", []):
        phone = item.get("phone", "")
        pnode = idx.get(_norm(phone)) if phone else None
        if pnode is None:
            skipped += 1
            continue
        p_outcome = item.get("outcome", "incorrect")
        p_conf = float(item.get("confidence", conf))
        b = twin.update(EvidenceObject(
            student_id=student_id, node_id=pnode, source="audio",
            outcome=p_outcome, confidence=p_conf,
            meta={"target_text": target_text, "phone": phone, "level": "phone"},
        ))
        phone_updates.append({"phone": phone, "node_id": pnode,
                              "outcome": p_outcome,
                              "mastery": round(b.mastery, 4)})
        # Contrastive L1: a production attempt on a confused phoneme moves the
        # substitution belief (error raises it; correct/mastery suppresses it).
        if l1 and l1 != "en":
            try:
                import l1_confusion as lc
                lc.note_production(student_id, l1, pnode,
                                   correct=(p_outcome == "correct"), confidence=p_conf)
                lc.suppress_on_mastery(student_id, l1, pnode, b.mastery)
            except Exception:
                pass
        # CASCADE: update the word↔phoneme edge beliefs this clip exercised.
        for ek in _word_phone_edges(node_id, pnode, graph_edges):
            eb = twin.update(EvidenceObject(
                student_id=student_id, node_id=pnode, edge_key=ek, source="audio",
                outcome=p_outcome, confidence=p_conf,
                meta={"target_text": target_text, "phone": phone, "level": "edge"},
            ))
            edge_updates.append({"edge_key": ek, "outcome": p_outcome,
                                 "edge_mastery": round(
                                     twin.get_edge(student_id, ek).mastery, 4)})

    # Causal diagnosis on a miss — reads the CASCADE edges just updated.
    diagnosis = None
    if outcome in ("incorrect", "partial"):
        try:
            import cause_net
            diagnosis = cause_net.diagnose(
                student_id, node_id, world, outcome=outcome, l1=l1).to_dict()
        except Exception:
            diagnosis = None

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
            "edge_updates": edge_updates,
            "diagnosis": diagnosis,
            "phones_unmapped": skipped}
