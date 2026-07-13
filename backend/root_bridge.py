"""Root Bridge — the curated Bengali↔English cognate table (real IE etymology).

Bengali (Indo-Aryan, via Sanskrit) and English (Germanic) are both Indo-European,
so a Bengali child ALREADY owns much of English's deep vocabulary. দন্ত is inside
দাঁত and inside *dental*; মাতা is *mother*; পঞ্চ is *penta*. This turns English
vocabulary from rote memorisation into REASONING: decode the word from a root you
already say.

HONESTY (the whole point): every entry is a genuine, accepted Indo-European
cognate from comparative linguistics — NOT a folk lookalike. So English *water*
bridges to উদক (Skt udaka, PIE *wódr̥), NEVER to জল (a real Bengali word but of
different, non-IE origin). A false cognate is a beautiful lie; we don't teach it.

Each entry:
  piece        the shared root/idea (roman)
  gloss        its meaning
  pie          reconstructed Proto-Indo-European root
  sanskrit      the Sanskrit form Bengali inherits
  bengali       Bengali word(s) the child already knows (script)
  bengali_roman transliteration
  english       the English word-family sharing the root (Germanic + Latin/Greek-derived)
  hook          plain-language reasoning bridge
  theme         for grouping
  note          caveat (literary form, false-friend warning) when needed

Sources: standard comparative Indo-European etymology (Sanskrit↔Latin↔Greek↔
Germanic correspondences; e.g. Mallory & Adams, Watkins). Curated, not generated.
No LLM.
"""

from __future__ import annotations

ROOTS: list[dict] = [
    # ── Family ──
    {"piece": "matr", "gloss": "mother", "pie": "*méh₂tēr", "sanskrit": "mātṛ",
     "bengali": "মাতা / মা", "bengali_roman": "mata / ma", "theme": "family",
     "english": ["mother", "maternal", "maternity", "matrix", "matron"],
     "hook": "You already say মা. English ‘mother’ is the very same ancient word."},
    {"piece": "pitr", "gloss": "father", "pie": "*ph₂tḗr", "sanskrit": "pitṛ",
     "bengali": "পিতা", "bengali_roman": "pita", "theme": "family",
     "english": ["father", "paternal", "paternity", "patron"],
     "hook": "পিতা and ‘father / paternal’ come from one word — p became f over time."},
    {"piece": "bhratr", "gloss": "brother", "pie": "*bʰréh₂tēr", "sanskrit": "bhrātṛ",
     "bengali": "ভ্রাতা / ভাই", "bengali_roman": "bhrata / bhai", "theme": "family",
     "english": ["brother", "fraternal", "fraternity"],
     "hook": "ভাই is ভ্রাতা — the same root as ‘brother’ and ‘fraternal’."},
    {"piece": "duhitr", "gloss": "daughter", "pie": "*dʰugh₂tḗr", "sanskrit": "duhitṛ",
     "bengali": "দুহিতা", "bengali_roman": "duhita", "theme": "family",
     "english": ["daughter"], "note": "দুহিতা is literary Bengali (common: মেয়ে).",
     "hook": "দুহিতা and ‘daughter’ are the same old word."},
    # ── Numbers ──
    {"piece": "dwi", "gloss": "two", "pie": "*dwóh₁", "sanskrit": "dvi",
     "bengali": "দুই / দ্বি", "bengali_roman": "dui / dwi", "theme": "number",
     "english": ["two", "dual", "duo", "di-", "dioxide", "bi-"],
     "hook": "দ্বি = two. So a ‘dioxide’ has দুই oxygens."},
    {"piece": "tri", "gloss": "three", "pie": "*tréyes", "sanskrit": "tri",
     "bengali": "তিন / ত্রি", "bengali_roman": "tin / tri", "theme": "number",
     "english": ["three", "tri-", "triangle", "trio", "trident"],
     "hook": "ত্রি = three. A ‘triangle’ is a ত্রি-cornered shape."},
    {"piece": "panca", "gloss": "five", "pie": "*pénkʷe", "sanskrit": "pañca",
     "bengali": "পাঁচ / পঞ্চ", "bengali_roman": "panch / pancha", "theme": "number",
     "english": ["five", "penta-", "pentagon"],
     "hook": "পঞ্চ = five. A ‘pentagon’ has পাঁচ sides."},
    {"piece": "dasa", "gloss": "ten", "pie": "*déḱm̥", "sanskrit": "daśa",
     "bengali": "দশ", "bengali_roman": "dosh", "theme": "number",
     "english": ["ten", "decimal", "decade", "deca-", "dime"],
     "hook": "দশ = ten. A ‘decade’ is দশ years; ‘decimal’ counts in দশ."},
    # ── Body ──
    {"piece": "danta", "gloss": "tooth", "pie": "*h₃dónts", "sanskrit": "danta",
     "bengali": "দন্ত / দাঁত", "bengali_roman": "danta / dat", "theme": "body",
     "english": ["tooth", "dental", "dentist", "denture"],
     "hook": "দাঁত is দন্ত — the same root inside ‘dental’ and ‘dentist’."},
    {"piece": "pad", "gloss": "foot", "pie": "*pód-", "sanskrit": "pāda",
     "bengali": "পদ / পা", "bengali_roman": "pod / pa", "theme": "body",
     "english": ["foot", "pedal", "pedestrian", "podium", "tripod"],
     "hook": "পা / পদ = foot. A ‘pedal’ is for your পা; a ‘tripod’ has ত্রি পদ."},
    {"piece": "hrd", "gloss": "heart", "pie": "*ḱḗr", "sanskrit": "hṛd",
     "bengali": "হৃদ / হৃদয়", "bengali_roman": "hrid / hridoy", "theme": "body",
     "english": ["heart", "cardiac", "cardiology"],
     "hook": "হৃদয় = heart; the same root gives ‘cardiac’."},
    {"piece": "manas", "gloss": "mind", "pie": "*men-", "sanskrit": "manas",
     "bengali": "মন / মনন", "bengali_roman": "mon / manan", "theme": "body",
     "english": ["mind", "mental", "mention", "mania"],
     "hook": "মন = mind. The same root is inside ‘mental’."},
    {"piece": "nas", "gloss": "nose", "pie": "*neh₂s-", "sanskrit": "nāsā",
     "bengali": "নাসা / নাক", "bengali_roman": "nasa / nak", "theme": "body",
     "english": ["nose", "nasal"],
     "hook": "নাক / নাসা = nose — the root inside ‘nasal’."},
    {"piece": "aksi", "gloss": "eye", "pie": "*h₃ekʷ-", "sanskrit": "akṣi",
     "bengali": "অক্ষি / আঁখি", "bengali_roman": "okkhi / ankhi", "theme": "body",
     "english": ["eye", "ocular", "optic"],
     "hook": "আঁখি / অক্ষি = eye — related to ‘ocular’."},
    {"piece": "prana", "gloss": "breath, life", "pie": "*h₂enh₁-", "sanskrit": "prāṇa",
     "bengali": "প্রাণ", "bengali_roman": "pran", "theme": "body",
     "english": ["animal", "animate", "anima"],
     "hook": "প্রাণ = breath/life. Latin ‘anima’ (breath) → ‘animal’ — a breathing thing."},
    # ── Nature ──
    {"piece": "udaka", "gloss": "water", "pie": "*wódr̥", "sanskrit": "udaka",
     "bengali": "উদক", "bengali_roman": "udak", "theme": "nature",
     "english": ["water", "hydro-", "hydrogen"],
     "note": "উদক is the TRUE cognate (literary). জল looks related but is NOT.",
     "hook": "উদক = water — the same root as ‘water’ and Greek ‘hydro’. (জল is a different, unrelated word.)"},
    {"piece": "surya", "gloss": "sun", "pie": "*sóh₂wl̥", "sanskrit": "sūrya",
     "bengali": "সূর্য", "bengali_roman": "surjo", "theme": "nature",
     "english": ["sun", "solar", "solstice"],
     "hook": "সূর্য = sun; the same root gives ‘solar’."},
    {"piece": "tara", "gloss": "star", "pie": "*h₂stḗr", "sanskrit": "tārā",
     "bengali": "তারা", "bengali_roman": "tara", "theme": "nature",
     "english": ["star", "astral", "astronomy", "stellar"],
     "hook": "তারা = star — the root inside ‘astronomy’ and ‘stellar’."},
    {"piece": "hima", "gloss": "snow, cold", "pie": "*ǵʰeim-", "sanskrit": "hima",
     "bengali": "হিম", "bengali_roman": "him", "theme": "nature",
     "english": ["hibernate", "hibernation"],
     "hook": "হিম = snow/cold (হিমালয় = হিম + আলয়, ‘abode of snow’). Latin gives ‘hibernate’ — to winter."},
    {"piece": "vayu", "gloss": "wind, air", "pie": "*h₂weh₁-", "sanskrit": "vāta / vāyu",
     "bengali": "বায়ু / বাত", "bengali_roman": "bayu / bat", "theme": "nature",
     "english": ["wind", "ventilate", "ventilation"],
     "hook": "বায়ু = air/wind; Latin ‘ventus’ (wind) → ‘ventilate’."},
    {"piece": "madhu", "gloss": "honey", "pie": "*médʰu", "sanskrit": "madhu",
     "bengali": "মধু", "bengali_roman": "modhu", "theme": "nature",
     "english": ["mead"],
     "hook": "মধু = honey — the same word as English ‘mead’ (honey-wine)."},
    # ── Animals ──
    {"piece": "go", "gloss": "cow", "pie": "*gʷṓus", "sanskrit": "go / gau",
     "bengali": "গো / গরু", "bengali_roman": "go / goru", "theme": "animal",
     "english": ["cow", "bovine", "beef"],
     "hook": "গরু / গো = cow — the same root as ‘bovine’."},
    {"piece": "svan", "gloss": "dog", "pie": "*ḱwṓ", "sanskrit": "śvan",
     "bengali": "শ্বান / সারমেয়", "bengali_roman": "shvan", "theme": "animal",
     "note": "শ্বান is literary (common: কুকুর).",
     "english": ["hound", "canine", "cynic"],
     "hook": "শ্বান = dog — related to ‘hound’ and ‘canine’."},
    {"piece": "hamsa", "gloss": "goose, swan", "pie": "*ǵʰh₂ens-", "sanskrit": "haṃsa",
     "bengali": "হংস / হাঁস", "bengali_roman": "hongsho / hnas", "theme": "animal",
     "english": ["goose"],
     "hook": "হাঁস / হংস = goose — the same old bird-word as ‘goose’."},
    {"piece": "mus", "gloss": "mouse", "pie": "*muh₂s", "sanskrit": "mūṣ",
     "bengali": "মূষিক", "bengali_roman": "mushik", "theme": "animal",
     "note": "মূষিক is literary (common: ইঁদুর).",
     "english": ["mouse", "muscle"],
     "hook": "মূষিক = mouse — the same root as ‘mouse’ (and ‘muscle’, a ‘little mouse’)."},
    # ── Knowing, doing ──
    {"piece": "jna", "gloss": "to know", "pie": "*ǵneh₃-", "sanskrit": "jñā",
     "bengali": "জ্ঞান / জানা", "bengali_roman": "gyan / jana", "theme": "action",
     "english": ["know", "diagnosis", "recognise", "gnostic"],
     "hook": "জানা / জ্ঞান = to know — the root inside ‘diagnosis’ and ‘recognise’."},
    {"piece": "vid", "gloss": "to see, know", "pie": "*weid-", "sanskrit": "vid / veda",
     "bengali": "বিদ্যা / বেদ", "bengali_roman": "bidya / bed", "theme": "action",
     "english": ["video", "vision", "wit", "idea", "evident"],
     "hook": "বিদ্যা = knowledge (from ‘to see’) — the root of ‘video’ and ‘vision’."},
    {"piece": "man-num", "gloss": "name", "pie": "*h₁nómn̥", "sanskrit": "nāman",
     "bengali": "নাম", "bengali_roman": "nam", "theme": "action",
     "english": ["name", "nominal", "nominate", "noun"],
     "hook": "নাম = name — the same word, and the root inside ‘nominate’."},
    {"piece": "sthā", "gloss": "to stand", "pie": "*steh₂-", "sanskrit": "sthā",
     "bengali": "স্থা / স্থির", "bengali_roman": "stha / sthir", "theme": "action",
     "english": ["stand", "status", "stable", "statue"],
     "hook": "স্থির = steady/standing — the root of ‘stand’, ‘status’, ‘stable’."},
    {"piece": "bhr", "gloss": "to bear, carry", "pie": "*bʰer-", "sanskrit": "bhṛ",
     "bengali": "ভর / বহন", "bengali_roman": "bhar / bahan", "theme": "action",
     "english": ["bear", "transfer", "fertile", "metaphor"],
     "hook": "ভর / বহন = to bear/carry — the root inside ‘transfer’ and ‘metaphor’."},
    {"piece": "da", "gloss": "to give", "pie": "*deh₃-", "sanskrit": "dā",
     "bengali": "দা / দান", "bengali_roman": "da / dan", "theme": "action",
     "english": ["donate", "donor", "data", "date"],
     "hook": "দান = giving — the root of ‘donate’ and ‘donor’."},
    {"piece": "yuga", "gloss": "yoke, join", "pie": "*yugóm", "sanskrit": "yuga / yoga",
     "bengali": "যোগ / যুগ", "bengali_roman": "jog / jug", "theme": "action",
     "english": ["yoke", "yoga", "junction", "conjugate", "join"],
     "hook": "যোগ = union/joining — the same root as ‘yoke’, ‘join’, ‘junction’."},
    {"piece": "vah", "gloss": "to carry", "pie": "*weǵʰ-", "sanskrit": "vah",
     "bengali": "বহন / বাহন", "bengali_roman": "bahan / bahon", "theme": "action",
     "english": ["vehicle", "wagon", "weigh"],
     "hook": "বাহন = a vehicle (something that carries) — the root of ‘vehicle’."},
    {"piece": "raj", "gloss": "king, rule", "pie": "*h₃rḗǵs", "sanskrit": "rājan",
     "bengali": "রাজা", "bengali_roman": "raja", "theme": "action",
     "english": ["regal", "royal", "rex", "regent"],
     "hook": "রাজা = king — the root of ‘regal’ and ‘royal’."},
    {"piece": "dvar", "gloss": "door", "pie": "*dʰwer-", "sanskrit": "dvāra",
     "bengali": "দ্বার", "bengali_roman": "dwar", "theme": "action",
     "english": ["door"],
     "hook": "দ্বার = door/gate — the same old word as ‘door’."},
    {"piece": "nau", "gloss": "boat", "pie": "*neh₂us", "sanskrit": "nau",
     "bengali": "নৌ / নৌকা", "bengali_roman": "nou / nouka", "theme": "action",
     "english": ["naval", "navy", "nautical", "navigate"],
     "hook": "নৌকা = boat — the root of ‘naval’ and ‘nautical’."},
    # ── more nature ──
    {"piece": "agni", "gloss": "fire", "pie": "*h₁n̥gʷnis", "sanskrit": "agni",
     "bengali": "অগ্নি", "bengali_roman": "ogni", "theme": "nature",
     "english": ["ignite", "ignition", "igneous"],
     "hook": "অগ্নি = fire — the Latin form ‘ignis’ gives ‘ignite’."},
    {"piece": "dhuma", "gloss": "smoke", "pie": "*dʰuh₂mós", "sanskrit": "dhūma",
     "bengali": "ধূম", "bengali_roman": "dhum", "theme": "nature",
     "english": ["fume", "fumigate", "perfume"],
     "hook": "ধূম = smoke — Latin ‘fumus’ gives ‘fume’ and ‘perfume’."},
    {"piece": "masa", "gloss": "month, moon", "pie": "*mḗh₁n̥s", "sanskrit": "māsa",
     "bengali": "মাস", "bengali_roman": "mas", "theme": "nature",
     "english": ["month", "menstrual", "semester"],
     "hook": "মাস = month — the same root as ‘month’ (a moon-cycle)."},
    {"piece": "asru", "gloss": "tear", "pie": "*dáḱru", "sanskrit": "aśru",
     "bengali": "অশ্রু", "bengali_roman": "osru", "theme": "nature",
     "english": ["tear"],
     "hook": "অশ্রু = tear — the same ancient word as English ‘tear’."},
    # ── more body ──
    {"piece": "asthi", "gloss": "bone", "pie": "*h₂ost-", "sanskrit": "asthi",
     "bengali": "অস্থি", "bengali_roman": "osthi", "theme": "body",
     "english": ["osteo-", "osteoporosis"],
     "hook": "অস্থি = bone — Greek ‘osteon’ gives ‘osteo-’ (as in osteoporosis)."},
    {"piece": "nakha", "gloss": "nail", "pie": "*h₃nogʷʰ-", "sanskrit": "nakha",
     "bengali": "নখ", "bengali_roman": "nokh", "theme": "body",
     "english": ["nail"],
     "hook": "নখ = nail — the same root as English ‘nail’."},
    {"piece": "bhru", "gloss": "eyebrow", "pie": "*h₃bʰruH", "sanskrit": "bhrū",
     "bengali": "ভ্রু", "bengali_roman": "bhru", "theme": "body",
     "english": ["brow", "eyebrow"],
     "hook": "ভ্রু = brow — the same word as English ‘brow’."},
    {"piece": "rudhira", "gloss": "blood, red", "pie": "*h₁rewdʰ-", "sanskrit": "rudhira",
     "bengali": "রুধির", "bengali_roman": "rudhir", "theme": "body",
     "english": ["red", "ruddy", "rust"],
     "hook": "রুধির = blood/red — the root of ‘red’ and ‘ruddy’."},
    # ── more action / quality ──
    {"piece": "madhya", "gloss": "middle", "pie": "*médʰyos", "sanskrit": "madhya",
     "bengali": "মধ্য", "bengali_roman": "moddho", "theme": "action",
     "english": ["middle", "medium", "mediate", "meso-"],
     "hook": "মধ্য = middle — the root of ‘medium’ and ‘middle’."},
    {"piece": "antara", "gloss": "inner, between", "pie": "*h₁énteros", "sanskrit": "antara",
     "bengali": "অন্তর", "bengali_roman": "ontor", "theme": "action",
     "english": ["interior", "internal", "enter", "entrance"],
     "hook": "অন্তর = inner — the root of ‘interior’ and ‘internal’."},
    {"piece": "purna", "gloss": "full", "pie": "*pleh₁-", "sanskrit": "pūrṇa",
     "bengali": "পূর্ণ", "bengali_roman": "purno", "theme": "action",
     "english": ["full", "plenty", "plural", "complete"],
     "hook": "পূর্ণ = full — the same root as ‘full’ and ‘plenty’."},
    {"piece": "guru", "gloss": "heavy", "pie": "*gʷréh₂us", "sanskrit": "guru",
     "bengali": "গুরু", "bengali_roman": "guru", "theme": "action",
     "english": ["gravity", "grave", "grief"],
     "hook": "গুরু = heavy (also ‘teacher’) — Latin ‘gravis’ gives ‘gravity’."},
    {"piece": "laghu", "gloss": "light, small", "pie": "*h₁léngʷʰ-", "sanskrit": "laghu",
     "bengali": "লঘু", "bengali_roman": "loghu", "theme": "action",
     "english": ["light", "levity", "elevate"],
     "hook": "লঘু = light (not heavy) — the same root as English ‘light’."},
    {"piece": "yuva", "gloss": "young", "pie": "*h₂yuh₁en-", "sanskrit": "yuvan",
     "bengali": "যুবা / যুবক", "bengali_roman": "juba / jubok", "theme": "action",
     "english": ["young", "juvenile", "junior"],
     "hook": "যুবা = young — the root of ‘young’ and ‘juvenile’."},
    {"piece": "mrtyu", "gloss": "death", "pie": "*mer-", "sanskrit": "mṛtyu",
     "bengali": "মৃত্যু / মৃত", "bengali_roman": "mrityu / mrito", "theme": "action",
     "english": ["mortal", "mortality", "murder"],
     "hook": "মৃত্যু = death; মৃত = dead — the root of ‘mortal’."},
    {"piece": "matra", "gloss": "measure", "pie": "*meh₁-", "sanskrit": "mātrā",
     "bengali": "মাত্রা", "bengali_roman": "matra", "theme": "action",
     "english": ["meter", "metre", "metric", "symmetry"],
     "hook": "মাত্রা = measure — Greek ‘metron’ gives ‘meter’ and ‘symmetry’."},
    {"piece": "smarana", "gloss": "to remember", "pie": "*(s)mer-", "sanskrit": "smṛti",
     "bengali": "স্মরণ / স্মৃতি", "bengali_roman": "smoron / smriti", "theme": "action",
     "english": ["memory", "memorial", "memorable"],
     "hook": "স্মৃতি = memory — the same root as ‘memory’ and ‘memorial’."},
    {"piece": "vak", "gloss": "speech, voice", "pie": "*wekʷ-", "sanskrit": "vāc / vākya",
     "bengali": "বাক্য / বাক", "bengali_roman": "bakko / bak", "theme": "action",
     "english": ["voice", "vocal", "vocabulary", "advocate"],
     "hook": "বাক্য = speech — Latin ‘vox’ gives ‘voice’ and ‘vocal’."},
    {"piece": "deva", "gloss": "god, shining", "pie": "*deiwos", "sanskrit": "deva",
     "bengali": "দেব", "bengali_roman": "deb", "theme": "action",
     "english": ["divine", "deity", "deify"],
     "hook": "দেব = god — Latin ‘deus’ gives ‘divine’ and ‘deity’."},
    {"piece": "jivana", "gloss": "life", "pie": "*gʷeyh₃-", "sanskrit": "jīva",
     "bengali": "জীবন / জীব", "bengali_roman": "jibon / jib", "theme": "action",
     "english": ["vivid", "vital", "survive", "revive"],
     "hook": "জীবন = life — Latin ‘vivere’ (to live) gives ‘vivid’ and ‘survive’."},
]


def _n(s: str) -> str:
    return (s or "").strip().lower()


def all_bridges() -> list[dict]:
    return ROOTS


def bridge_for_word(word: str) -> dict | None:
    """Find the cognate bridge for an English word — FULL-WORD match only.
    Prefix forms like 'di-'/'tri-' are display-only and never used to match, so
    'diagnosis' (dia-gnosis) can't be misread as 'di' (two). No false cognates."""
    w = _n(word)
    if not w:
        return None
    ws = w[:-1] if w.endswith("s") else w              # crude singular
    for e in ROOTS:
        for en in e["english"]:
            enl = _n(en)
            if enl.endswith("-") or len(enl) < 3:       # prefix / too-short → display only
                continue
            base = enl[:-1] if enl.endswith("s") else enl
            if w == enl or ws == base:
                return e
    return None


def by_theme() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for e in ROOTS:
        out.setdefault(e.get("theme", "other"), []).append(e)
    return out
