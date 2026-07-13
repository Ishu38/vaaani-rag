"""Transcription engine — the phonetic-literacy rung of CASCADE.

Neil's vision: after a few days a child should read any dictionary and WRITE the
IPA of any word with confidence. That needs a word→IPA reference for arbitrary
words (not just the ~26 graph phonemes). espeak-ng supplies it:

    espeak-ng --ipa -q "dictionary"  →  dˈɪkʃənəɹi

Two skills sit on top of that reference:
  • READ the dictionary — reveal a word's IPA, learner says it (the ear checks).
  • WRITE the notation — learner types the IPA; we align it against the espeak
    reference phoneme-by-phoneme, score it, and fold the outcomes into the SAME
    machinery as the ear: phoneme-node BKT, the contrastive L1 confusion field,
    and the cause-net. So transcription practice moves the learner's CASCADE
    state exactly like pronunciation does.

CPU-only, deterministic, no LLM. espeak placement is the ear's — present on the
VM; ipa_for() returns None when unavailable and callers degrade gracefully.
"""

from __future__ import annotations

import re
import subprocess

# multi-character IPA units to keep whole during tokenisation
_CLUSTERS = ["t͡ʃ", "d͡ʒ", "tʃ", "dʒ", "aɪ", "aʊ", "ɔɪ", "eɪ", "oʊ", "əʊ",
             "ɪə", "eə", "ʊə", "ɛə"]
_STRESS = "ˈˌ.|ˑ"                                   # stress / syllable marks to drop
_LENGTH = "ː"


def ipa_for(word: str) -> str | None:
    """Reference IPA for a word via espeak-ng. None if espeak is unavailable."""
    word = (word or "").strip()
    if not word or not re.fullmatch(r"[A-Za-z'\- ]{1,40}", word):
        return None
    try:
        r = subprocess.run(["espeak-ng", "--ipa", "-q", word],
                           capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    out = (r.stdout or "").strip()
    return out or None


def tokenize(ipa: str) -> list[str]:
    """Split an IPA string into phoneme tokens (affricates/diphthongs kept whole,
    length attached to the vowel, stress/syllable marks dropped)."""
    s = re.sub(f"[{_STRESS}]", "", ipa or "").replace(" ", "")
    tokens: list[str] = []
    i = 0
    while i < len(s):
        matched = None
        for cl in _CLUSTERS:
            if s.startswith(cl, i):
                matched = cl
                break
        if matched:
            tokens.append(matched); i += len(matched)
        else:
            tokens.append(s[i]); i += 1
        # attach a following length mark to the token just added
        if i < len(s) and s[i] == _LENGTH:
            tokens[-1] += _LENGTH; i += 1
    return [t for t in tokens if t.strip()]


def _align(ref: list[str], hyp: list[str]) -> list[tuple[str, str]]:
    """Needleman-Wunsch alignment → list of (ref_tok|'', hyp_tok|'') pairs."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): d[i][0] = i
    for j in range(1, m + 1): d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    i, j, out = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            out.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            out.append((ref[i - 1], "")); i -= 1
        else:
            out.append(("", hyp[j - 1])); j -= 1
    out.reverse()
    return out


def check(student_ipa: str, reference_ipa: str) -> dict:
    """Compare a learner's typed IPA to the reference, phoneme by phoneme."""
    ref = tokenize(reference_ipa)
    hyp = tokenize(student_ipa)
    pairs = _align(ref, hyp)
    per_phoneme = []
    correct = 0
    for r, h in pairs:
        if r and h and r == h:
            outcome = "correct"; correct += 1
        elif r and h:
            outcome = "substituted"
        elif r and not h:
            outcome = "missed"
        else:
            outcome = "inserted"
        per_phoneme.append({"ref": r, "got": h, "outcome": outcome})
    score = correct / len(ref) if ref else 0.0
    return {"score": round(score, 3), "n_ref": len(ref),
            "reference_tokens": ref, "per_phoneme": per_phoneme}


def ingest_transcription(student_id: str, word: str, student_ipa: str,
                         world, l1: str = "en") -> dict:
    """Score a written transcription and fold it into CASCADE + confusion +
    cause-net — the same ingress the ear uses, so writing IPA moves the learner
    state like speaking does. Returns a summary the SPA can show."""
    reference = ipa_for(word)
    if reference is None:
        return {"recorded": False, "reason": "transcription reference unavailable"}

    import cognitive_twin as twin
    from evidence_graph import EvidenceObject
    from ear import phone_node_index, _norm

    result = check(student_ipa, reference)
    idx = phone_node_index(world)

    phone_updates, skipped = [], 0
    for item in result["per_phoneme"]:
        ref = item["ref"]
        if not ref:                                   # inserted extra — no target
            continue
        pnode = idx.get(_norm(ref.replace("ː", "")))
        if pnode is None:
            skipped += 1
            continue
        correct = item["outcome"] == "correct"
        b = twin.update(EvidenceObject(
            student_id=student_id, node_id=pnode, source="transcription",
            outcome="correct" if correct else "incorrect", confidence=0.9,
            meta={"word": word, "level": "phone", "ref": ref, "got": item["got"]}))
        phone_updates.append({"ref": ref, "node_id": pnode,
                              "outcome": item["outcome"],
                              "mastery": round(b.mastery, 4)})
        # contrastive L1: writing a confused phoneme moves the substitution belief
        if l1 and l1 != "en":
            try:
                import l1_confusion as lc
                lc.note_production(student_id, l1, pnode, correct=correct, confidence=0.9)
                lc.suppress_on_mastery(student_id, l1, pnode, b.mastery)
            except Exception:
                pass

    outcome = ("correct" if result["score"] >= 0.8
               else "partial" if result["score"] >= 0.5 else "incorrect")
    diagnosis = None
    if outcome in ("incorrect", "partial") and phone_updates:
        try:
            import cause_net
            worst = min(phone_updates, key=lambda p: p["mastery"])
            diagnosis = cause_net.diagnose(student_id, worst["node_id"], world,
                                           outcome=outcome, l1=l1).to_dict()
        except Exception:
            diagnosis = None

    return {"recorded": True, "word": word, "reference": reference,
            "score": result["score"], "outcome": outcome,
            "per_phoneme": result["per_phoneme"],
            "phone_updates": phone_updates, "phones_unmapped": skipped,
            "diagnosis": diagnosis}
