"""Morphology — the solid, substantial word-building ground for BOTH languages.

The Root Bridge (root_bridge.py) gives shared cognate ROOTS. This module gives
the shared MACHINERY: both Bengali (via Sanskrit) and English build words as
  [ prefix ] + root + [ suffix ]
and, being Indo-European, several affixes are literally COGNATE. So a Bengali
child decodes English not by memorising words but by reasoning with pieces they
already own.

Contents:
  COGNATE_AFFIXES  Bengali উপসর্গ ↔ English (Latin/Greek) prefix — real IE cognates only.
  EN_PREFIXES      English prefixes (decoding inventory) + meanings.
  EN_SUFFIXES      English suffixes / combining forms (many are Greek roots).
  EN_ROOTS         common Latin/Greek roots in English.
  BN_UPASARGA      Sanskrit উপসর্গ used in Bengali (standard 20) + meaning + example.
  BN_PRATYAYA      common Bengali প্রত্যয় (suffixes).

Grounding: Sanskrit উপসর্গ set from standard Bengali ব্যাকরণ; cognate affix pairs
from comparative Indo-European (PIE prefixes *pro, *per, *sem, *upo, *apo);
English roots/affixes from standard morphology + Eide, Logic of English.
Curated, conservative (only well-attested cognates); no LLM.
"""

from __future__ import annotations

# ── Cognate affixes: the same prefix in both languages (real IE) ──
COGNATE_AFFIXES: list[dict] = [
    {"bn": "প্র", "bn_roman": "pra", "en": "pro", "pie": "*pro", "gloss": "forward, before",
     "bn_ex": "প্রগতি (progress)", "en_ex": ["progress", "produce", "project", "promote"],
     "hook": "প্র = forward. প্রগতি is ‘forward-going’ — English ‘progress’ is the same."},
    {"bn": "পরি", "bn_roman": "pari", "en": "peri", "pie": "*per", "gloss": "around, all-round",
     "bn_ex": "পরিভ্রমণ (going around)", "en_ex": ["perimeter", "periphery", "period", "periscope"],
     "hook": "পরি = around. পরিভ্রমণ is going all-around — a ‘perimeter’ is the all-around edge."},
    {"bn": "সম্", "bn_roman": "sam", "en": "syn / sym / com", "pie": "*sem", "gloss": "together, with",
     "bn_ex": "সংগঠন (organising together)", "en_ex": ["synthesis", "sympathy", "combine", "symphony"],
     "hook": "সম্ = together. সংগীত (sounds together) — English ‘symphony’ is ‘sounds together’ too."},
    {"bn": "উপ", "bn_roman": "upa", "en": "hypo / sub", "pie": "*upo", "gloss": "under, near, lesser",
     "bn_ex": "উপনগর (sub-city, suburb)", "en_ex": ["hypothesis", "hypodermic", "suburb", "subway"],
     "hook": "উপ = under/near. উপনগর is a ‘sub-city’ — the same idea as ‘suburb’."},
    {"bn": "অপ", "bn_roman": "apa", "en": "apo / ab", "pie": "*apo", "gloss": "away, off",
     "bn_ex": "অপসারণ (removing away)", "en_ex": ["apology", "apostrophe", "absent", "abnormal"],
     "hook": "অপ = away. অপসারণ is moving away — the same root is in ‘apo’-strophe and ‘ab’-sent."},
    {"bn": "অতি", "bn_roman": "ati", "en": "—", "pie": "*eti", "gloss": "beyond, excess",
     "bn_ex": "অতিরিক্ত (excess)", "en_ex": [],
     "note": "অতি has no everyday English cognate prefix; nearest sense is ‘extra/ultra/hyper’."},
]

# ── English prefixes (decoding inventory) ──
EN_PREFIXES: dict[str, dict] = {
    "un":    {"gloss": "not, opposite", "ex": "unhappy"},
    "re":    {"gloss": "again, back", "ex": "return"},
    "dis":   {"gloss": "not, apart", "ex": "disconnect"},
    "in/im": {"gloss": "not / into", "ex": "invisible, import"},
    "pre":   {"gloss": "before", "ex": "preview"},
    "post":  {"gloss": "after", "ex": "postpone"},
    "trans": {"gloss": "across", "ex": "transport"},
    "inter": {"gloss": "between", "ex": "international"},
    "super": {"gloss": "above, over", "ex": "supervise"},
    "sub":   {"gloss": "under", "ex": "submarine"},
    "anti":  {"gloss": "against", "ex": "antibiotic"},
    "auto":  {"gloss": "self", "ex": "automatic"},
    "tele":  {"gloss": "far", "ex": "telephone"},
    "micro": {"gloss": "small", "ex": "microscope"},
    "mono":  {"gloss": "one", "ex": "monorail"},
    "bi":    {"gloss": "two", "ex": "bicycle"},
    "multi": {"gloss": "many", "ex": "multiply"},
}

# ── English suffixes / combining forms (many are Greek roots themselves) ──
EN_SUFFIXES: dict[str, dict] = {
    "graph": {"gloss": "writing, drawing", "ex": "photograph", "greek": "graphein"},
    "phone": {"gloss": "sound, voice", "ex": "telephone", "greek": "phone"},
    "scope": {"gloss": "look, see", "ex": "microscope", "greek": "skopein"},
    "meter": {"gloss": "measure", "ex": "thermometer", "greek": "metron"},
    "ology": {"gloss": "study of", "ex": "biology", "greek": "logos"},
    "tion":  {"gloss": "act / state of", "ex": "action"},
    "ity":   {"gloss": "quality of", "ex": "purity"},
    "ism":   {"gloss": "belief / system", "ex": "realism"},
    "ist":   {"gloss": "person who", "ex": "artist"},
    "able":  {"gloss": "can be", "ex": "readable"},
    "ful":   {"gloss": "full of", "ex": "helpful"},
    "less":  {"gloss": "without", "ex": "fearless"},
    "ness":  {"gloss": "state of", "ex": "kindness"},
    "ment":  {"gloss": "result of", "ex": "movement"},
}

# ── Common Latin/Greek roots in English (decode the middle of the word) ──
EN_ROOTS: dict[str, dict] = {
    "bio":    {"gloss": "life", "ex": "biology"},
    "geo":    {"gloss": "earth", "ex": "geography"},
    "photo":  {"gloss": "light", "ex": "photograph"},
    "therm":  {"gloss": "heat", "ex": "thermometer"},
    "aqua":   {"gloss": "water", "ex": "aquarium"},
    "port":   {"gloss": "carry", "ex": "transport"},
    "dict":   {"gloss": "say", "ex": "dictionary"},
    "spect":  {"gloss": "look", "ex": "inspect"},
    "struct": {"gloss": "build", "ex": "construct"},
    "vid/vis":{"gloss": "see", "ex": "video, vision"},
    "aud":    {"gloss": "hear", "ex": "audio"},
    "man":    {"gloss": "hand", "ex": "manual"},
    "ped/pod":{"gloss": "foot", "ex": "pedal, tripod"},
    "chrono": {"gloss": "time", "ex": "chronology"},
}

# ── Bengali উপসর্গ (Sanskrit-derived, standard set) ──
BN_UPASARGA: list[dict] = [
    {"bn": "প্র", "roman": "pra", "gloss": "forward, intense", "ex": "প্রগতি, প্রবল"},
    {"bn": "পরা", "roman": "para", "gloss": "away, opposite", "ex": "পরাজয়, পরাক্রম"},
    {"bn": "অপ", "roman": "apa", "gloss": "away, bad", "ex": "অপমান, অপকার"},
    {"bn": "সম্", "roman": "sam", "gloss": "well, together", "ex": "সংগীত, সম্মান"},
    {"bn": "অনু", "roman": "anu", "gloss": "after, following", "ex": "অনুগামী, অনুসরণ"},
    {"bn": "অব", "roman": "aba", "gloss": "down, off", "ex": "অবনতি, অবতরণ"},
    {"bn": "নির্/নিস্", "roman": "nir/nis", "gloss": "without, out", "ex": "নির্ভয়, নিস্তেজ"},
    {"bn": "দুর্/দুস্", "roman": "dur/dus", "gloss": "bad, difficult", "ex": "দুর্বল, দুষ্কর"},
    {"bn": "বি", "roman": "bi", "gloss": "special, apart", "ex": "বিশেষ, বিদেশ"},
    {"bn": "অধি", "roman": "adhi", "gloss": "over, above", "ex": "অধিপতি, অধিকার"},
    {"bn": "সু", "roman": "su", "gloss": "good, well", "ex": "সুন্দর, সুগম"},
    {"bn": "উৎ", "roman": "ut", "gloss": "up, upward", "ex": "উন্নতি, উৎসাহ"},
    {"bn": "পরি", "roman": "pari", "gloss": "around, complete", "ex": "পরিভ্রমণ, পরিপূর্ণ"},
    {"bn": "প্রতি", "roman": "prati", "gloss": "towards, each, against", "ex": "প্রতিদিন, প্রতিধ্বনি"},
    {"bn": "অতি", "roman": "ati", "gloss": "excess, beyond", "ex": "অতিরিক্ত, অত্যধিক"},
    {"bn": "অভি", "roman": "abhi", "gloss": "towards", "ex": "অভিমুখ, অভিযান"},
    {"bn": "উপ", "roman": "upa", "gloss": "near, sub", "ex": "উপনগর, উপদেশ"},
    {"bn": "আ", "roman": "a", "gloss": "until, slightly", "ex": "আগমন, আহরণ"},
]

# ── Common Bengali প্রত্যয় (suffixes forming nouns/adjectives) ──
BN_PRATYAYA: list[dict] = [
    {"bn": "-তা", "roman": "-ta", "gloss": "-ness (quality)", "ex": "সুন্দরতা, মধুরতা"},
    {"bn": "-ত্ব", "roman": "-twa", "gloss": "-ness/-hood (state)", "ex": "মনুষ্যত্ব, গুরুত্ব"},
    {"bn": "-ইক", "roman": "-ik", "gloss": "-ic/-al (relating to)", "ex": "শারীরিক, সামাজিক"},
    {"bn": "-বান/-মান", "roman": "-ban/-man", "gloss": "-ful (having)", "ex": "ধনবান, বুদ্ধিমান"},
    {"bn": "-হীন", "roman": "-hin", "gloss": "-less (without)", "ex": "ভয়হীন, জলহীন"},
    {"bn": "-কার", "roman": "-kar", "gloss": "-er (one who does)", "ex": "চিত্রকার, গল্পকার"},
    {"bn": "-ময়", "roman": "-moy", "gloss": "-ful of / made of", "ex": "জলময়, আনন্দময়"},
    {"bn": "-আলি", "roman": "-ali", "gloss": "-ness / collective", "ex": "মিতালি, ঘটকালি"},
    {"bn": "-অন", "roman": "-on", "gloss": "act of (verbal noun)", "ex": "গমন, দর্শন"},
    {"bn": "-ইত", "roman": "-ito", "gloss": "-ed (done, having)", "ex": "চিন্তিত, আনন্দিত"},
]


def _n(s: str) -> str:
    return (s or "").strip().lower()


def cognate_prefix(word: str) -> dict | None:
    """If an English word starts with a cognate prefix, return the bridge."""
    w = _n(word)
    for a in COGNATE_AFFIXES:
        for en in a["en"].replace(" ", "").split("/"):
            if en and en != "—" and w.startswith(en) and len(w) > len(en) + 1:
                return a
    return None


def decode(word: str) -> dict:
    """Best-effort decomposition of an English word into known pieces."""
    w = _n(word)
    pref = next((p for p in EN_PREFIXES
                 if any(w.startswith(x) for x in p.split("/"))), None)
    suf = next((s for s in EN_SUFFIXES if w.endswith(s)), None)
    root = next((r for r in EN_ROOTS if r.split("/")[0] in w), None)
    # don't mis-split bio+logy as bi+ology: if a root at the start swallows the
    # prefix (bio ⊃ bi), the root wins.
    if pref and root:
        rk = root.split("/")[0]
        if w.startswith(rk) and rk.startswith(pref.split("/")[0]):
            pref = None
    cog = cognate_prefix(word)
    return {"word": word, "prefix": pref, "root": root, "suffix": suf,
            "cognate_prefix": cog}
