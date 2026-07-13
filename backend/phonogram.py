"""Phonograms — WHY English spells one sound many ways (Eide, Logic of English).

The mirror of the L1-script bridge. Bengali is transparent: শ is always /ʃ/, /ʃ/
is always শ — one letter, one sound. English is deep: the /ʃ/ sound is written
sh, ti, ci, ss, ch; and one letter (s) says /s/ AND /z/. Denise Eide's insight
is that this is not chaos but a system of ~74 phonograms — "pictures of sounds".

This module maps each ENGLISH SOUND to its common spellings (with a familiar
example), so a vernacular-medium child sees, from first principles, why English
writes one sound many ways — and that their own script does not. Grounded in
"Uncovering the Logic of English" (Eide, 2012); common spellings first.
No LLM; static reference.
"""

from __future__ import annotations

# IPA sound (espeak symbols) → [(spelling, example word), ...], common first.
PHONOGRAM: dict[str, list[tuple[str, str]]] = {
    # consonants that English spells several ways
    "f":  [("f", "fish"), ("ph", "phone"), ("gh", "laugh"), ("ff", "off")],
    "ʃ":  [("sh", "ship"), ("ti", "nation"), ("ci", "special"), ("ss", "mission"), ("ch", "chef")],
    "tʃ": [("ch", "chip"), ("tch", "watch")],
    "dʒ": [("j", "jam"), ("g", "gem"), ("dge", "bridge")],
    "k":  [("c", "cat"), ("k", "kite"), ("ck", "back"), ("ch", "school"), ("qu", "queen")],
    "s":  [("s", "sun"), ("c", "city"), ("ss", "miss")],
    "z":  [("z", "zoo"), ("s", "is"), ("se", "rose")],
    "ŋ":  [("ng", "sing"), ("n", "think")],
    "θ":  [("th", "thin")],
    "ð":  [("th", "this")],
    "j":  [("y", "yes"), ("i", "onion")],
    "w":  [("w", "wet"), ("wh", "what")],
    "r":  [("r", "run"), ("wr", "write"), ("rr", "carry")],
    "m":  [("m", "man"), ("mm", "hammer")],
    "n":  [("n", "net"), ("kn", "knee"), ("nn", "dinner")],
    "l":  [("l", "leg"), ("ll", "ball")],
    "g":  [("g", "go"), ("gg", "egg")],
    "b":  [("b", "bat")], "d": [("d", "dog")], "p": [("p", "pan")],
    "t":  [("t", "top"), ("tt", "letter"), ("ed", "jumped")],
    "v":  [("v", "van")], "h": [("h", "hat")],
    # vowels — where English spelling really blooms
    "iː": [("ee", "see"), ("ea", "sea"), ("e", "me"), ("y", "happy"), ("ey", "key"), ("ie", "field")],
    "ɪ":  [("i", "sit"), ("y", "gym")],
    "eɪ": [("a", "table"), ("ai", "rain"), ("ay", "day"), ("ei", "vein")],
    "aɪ": [("i", "find"), ("y", "my"), ("igh", "night"), ("ie", "pie")],
    "əʊ": [("o", "go"), ("oa", "boat"), ("ow", "snow"), ("oe", "toe")],
    "oʊ": [("o", "go"), ("oa", "boat"), ("ow", "snow")],
    "uː": [("oo", "moon"), ("u", "ruby"), ("ew", "new"), ("ui", "fruit")],
    "ʊ":  [("oo", "book"), ("u", "put")],
    "ɔː": [("or", "for"), ("au", "sauce"), ("aw", "saw"), ("augh", "caught")],
    "ɑː": [("ar", "car"), ("a", "father")],
    "ɜː": [("er", "her"), ("ir", "bird"), ("ur", "turn")],
    "ə":  [("a", "about"), ("er", "water"), ("o", "lemon")],
    "æ":  [("a", "cat")],
    "e":  [("e", "bed"), ("ea", "bread")],
    "ɛ":  [("e", "bed"), ("ea", "bread")],
    "ʌ":  [("u", "cup"), ("o", "son")],
    "ɒ":  [("o", "dog"), ("a", "watch")],
    "ɔ":  [("o", "dog"), ("aw", "saw")],
    "aʊ": [("ow", "cow"), ("ou", "out")],
    "ɔɪ": [("oy", "boy"), ("oi", "coin")],
    "ɪə": [("ear", "ear"), ("eer", "deer")],
    "eə": [("air", "hair"), ("are", "care")],
}


def _norm(ipa: str) -> str:
    return (ipa or "").replace("ˈ", "").replace("ˌ", "").strip()


def spellings_for(ipa: str) -> list[dict]:
    """Common English spellings of a sound, with an example each."""
    key = _norm(ipa)
    entry = PHONOGRAM.get(key) or PHONOGRAM.get(key.replace("ː", ""))
    if not entry:
        return []
    return [{"spelling": s, "example": w} for s, w in entry]


def is_multi_spelled(ipa: str) -> bool:
    return len(spellings_for(ipa)) > 1
