"""L1-script bridge — read any English word in your OWN letters.

A vernacular-medium child (Bangla / Hindi / Tamil) is fluently literate in a
PHONEMIC script — each akshara reliably maps to a sound. That is the bridge:
anchor every English sound to a letter the child already reads and writes, so
they never leave their mother tongue to reach English. IPA is the grown-up name
earned last; the akshara is the first notation.

Design (the architecture makes it consistent):
  • For a sound the L1 HAS — the exact akshara (status "known").
  • For a sound the L1 LACKS — the nearest akshara is the CONFUSION ATTRACTOR
    (the sound they'd substitute), flagged "new". So l1_script and l1_confusion
    are one system: the gaps here are exactly the confusion targets there.

Curated, honest — common consonants + vowels first; Tamil voicing is positional
(one akshara for a voiced/voiceless pair) so those are flagged. Grounded, not
fetched; no LLM.
"""

from __future__ import annotations

from l1_confusion import attractor_for

# Consonant aksharas for sounds each L1 HAS (phoneme-node → akshara).
KNOWN: dict[str, dict[str, str]] = {
    # Bengali — grounded in Vidyasagar's বর্ণপরিচয় (Barnaparichay). ফ is Bengali's
    # letter for /f/; দন্ত্য ত is the canonical "t"; শ/ষ/স all lean /ʃ/ (স→ʃ tendency
    # lives in the confusion field, not here).
    "bn": {"phoneme-p": "প", "phoneme-b": "ব", "phoneme-t": "ত", "phoneme-d": "দ",
           "phoneme-k": "ক", "phoneme-g": "গ", "phoneme-m": "ম", "phoneme-n": "ন",
           "phoneme-ng": "ঙ", "phoneme-l": "ল", "phoneme-r": "র", "phoneme-h": "হ",
           "phoneme-s": "স", "phoneme-sh": "শ", "phoneme-ch": "চ", "phoneme-j": "জ",
           "phoneme-y": "য়", "phoneme-f": "ফ"},
    "hi": {"phoneme-p": "प", "phoneme-b": "ब", "phoneme-t": "त", "phoneme-d": "द",
           "phoneme-k": "क", "phoneme-g": "ग", "phoneme-m": "म", "phoneme-n": "न",
           "phoneme-ng": "ङ", "phoneme-l": "ल", "phoneme-r": "र", "phoneme-h": "ह",
           "phoneme-s": "स", "phoneme-sh": "श", "phoneme-ch": "च", "phoneme-j": "ज",
           "phoneme-y": "य", "phoneme-w": "व"},
    "ta": {"phoneme-p": "ப", "phoneme-b": "ப", "phoneme-t": "த", "phoneme-d": "த",
           "phoneme-k": "க", "phoneme-g": "க", "phoneme-m": "ம", "phoneme-n": "ந",
           "phoneme-ng": "ங", "phoneme-l": "ல", "phoneme-r": "ர", "phoneme-h": "ஹ",
           "phoneme-s": "ஸ", "phoneme-sh": "ஷ", "phoneme-ch": "ச", "phoneme-j": "ஜ",
           "phoneme-y": "ய", "phoneme-w": "வ"},
}

# Nearest independent vowel letter for the common English vowels (espeak IPA).
VOWELS: dict[str, dict[str, str]] = {
    # Bengali (Barnaparichay স্বরবর্ণ): no length distinction in SOUND, so English
    # length is taught through the long LETTERS — sheep ঈ vs ship ই, pool ঊ vs pull উ.
    # অ = /ɔ/ (inherent vowel), so ʌ/ɔ/ɒ/ə anchor to অ; আ = /a/.
    "bn": {"iː": "ঈ", "i": "ই", "ɪ": "ই", "e": "এ", "ɛ": "এ", "æ": "অ্যা",
           "a": "আ", "ɑ": "আ", "ɑː": "আ", "ʌ": "অ", "ɒ": "অ", "ɔ": "অ", "ɔː": "অ",
           "o": "ও", "oː": "ও", "uː": "ঊ", "u": "উ", "ʊ": "উ", "ə": "অ", "ɜ": "অ", "ɜː": "অ",
           "əʊ": "ও", "oʊ": "ও", "aɪ": "আই", "aʊ": "আউ", "eɪ": "এই", "ɔɪ": "অই"},
    "hi": {"i": "इ", "ɪ": "इ", "e": "ए", "ɛ": "ऐ", "æ": "ऐ", "a": "अ", "ɑ": "आ",
           "ʌ": "अ", "ɒ": "ऑ", "ɔ": "ओ", "o": "ओ", "ʊ": "उ", "u": "उ", "ə": "अ", "ɜ": "अ",
           "əʊ": "ओ", "oʊ": "ओ", "aɪ": "आइ", "aʊ": "आउ", "eɪ": "एइ", "ɔɪ": "ऑइ"},
    "ta": {"i": "இ", "ɪ": "இ", "e": "எ", "ɛ": "எ", "æ": "ஆ", "a": "அ", "ɑ": "ஆ",
           "ʌ": "அ", "ɒ": "ஒ", "ɔ": "ஓ", "o": "ஓ", "ʊ": "உ", "u": "உ", "ə": "அ", "ɜ": "அ",
           "əʊ": "ஓ", "oʊ": "ஓ", "aɪ": "ஐ", "aʊ": "ஔ", "eɪ": "ஏஇ", "ɔɪ": "ஒஇ"},
}

# espeak IPA quirks → graph phoneme display symbols
IPA_ALIAS = {"ɹ": "r", "ɡ": "g", "ɜ": "ə"}

# Voiced consonants Tamil script does not distinguish from their voiceless pair.
_TA_VOICING_NEW = {"phoneme-b", "phoneme-d", "phoneme-g"}


def supported(l1: str) -> bool:
    return l1 in KNOWN


def akshara_for_consonant(l1: str, node: str) -> tuple[str | None, str]:
    """(akshara, status) for a consonant phoneme node.
    status ∈ {known, new, voicing-new}. 'new' means the L1 lacks the sound and
    the akshara shown is the confusion attractor (what they'd substitute)."""
    table = KNOWN.get(l1, {})
    if node in table:
        if l1 == "ta" and node in _TA_VOICING_NEW:
            return table[node], "voicing-new"
        return table[node], "known"
    att = attractor_for(l1, node)               # gap → nearest = confusion attractor
    if att and att[0] in table:
        return table[att[0]], "new"
    return None, "new"


def _norm_vowel(tok: str) -> str:
    return tok.replace("ː", "").replace("ˈ", "").strip()


def word_in_script(word: str, l1: str, world) -> dict | None:
    """Render an English word sound-by-sound in the child's own letters.
    Returns per-sound anchors (+ new-sound flags) and a joined script string."""
    if not supported(l1):
        return None
    import transcribe as tr
    from ear import phone_node_index, _norm
    ref = tr.ipa_for(word)
    if ref is None:
        return None
    idx = phone_node_index(world)
    vmap = VOWELS.get(l1, {})

    sounds, script_parts, new_sounds = [], [], []
    for tok in tr.tokenize(ref):
        base = _norm(tok.replace("ː", ""))
        node = idx.get(IPA_ALIAS.get(base, base))
        if node:                                        # consonant in the graph
            aksh, status = akshara_for_consonant(l1, node)
            ipa_disp = world.display(node)
        else:                                           # vowel (or unmapped)
            aksh = vmap.get(tok) or vmap.get(_norm_vowel(tok))
            status = "vowel" if aksh else "other"
            ipa_disp = tok
        sounds.append({"ipa": ipa_disp, "node": node, "akshara": aksh, "status": status})
        if aksh:
            script_parts.append(aksh)
        if status in ("new", "voicing-new"):
            new_sounds.append({"ipa": ipa_disp, "akshara": aksh, "status": status})

    return {"word": word, "l1": l1, "ipa": ref,
            "script": "".join(script_parts), "sounds": sounds,
            "new_sounds": new_sounds}
