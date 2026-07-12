#!/usr/bin/env python3
"""Phase 1 — Structural Linguistics Graph Seeder.

Enriches the existing word-root graph with the full Vaaani language-reasoning
hierarchy.  Every node type in the architecture is materialised from the
vetted spiral curriculum (units_spiral.json), with no LLM calls:

  root     — already built by build_library_graph.py
  word     — already built
  prefix   — extracted from root pieces (BI, TRI, UNI, …)
  meaning  — one node per unique root/prefi x meaning
  phoneme  — curated 24-cardinal-phoneme set with sensory nicknames
  grapheme — common English graphemes (f, ph, sh, ch, th, ng, …)
  alias    — sensory / alternative names that resolve to the same node
  language — English / Hindi / Bangla / Sanskrit anchor nodes

Edge types:
  means           root → meaning (BI → "two")
  used_in         REVERSE of root_of (BI → bicycle, bilingual, …)
  prerequisite_for DISCOVERY edge (family word → new word)   
  written_as      phoneme → grapheme (/f/ → f, ph)
  sounds_like     word → phoneme (phone → /f/)
  alias_of        alias node → canonical node
  cognate_with    English root ↔ cross-lingual counterpart

Metadata on every node: grade, display_name, descriptions, chunk_ids=[-1].

Idempotent — safe to run any number of times.  Use after
build_library_graph.py and before building communities.

Run:  cd backend && python graph/seeder.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import GRAPH_PATH, COMMUNITIES_PATH
from graph import KnowledgeGraph, normalize
from extractor import Entity, Relation
from community import Community, load_communities, save_communities

SPIRAL = pathlib.Path(
    os.environ.get("VAAANI_SPIRAL_PATH", "")
) if os.environ.get("VAAANI_SPIRAL_PATH") else None

_SPIRAL_CANDIDATES = [
    os.path.expanduser("~/vaaani-model/dataset/units_spiral.json"),
    "/home/iamanushka32/vaaani-model/dataset/units_spiral.json",
    "/home/ishu/vaaani-model/dataset/units_spiral.json",
    os.path.expanduser("~/Desktop/rag-assistant/../vaaani-model/dataset/units_spiral.json"),
]

if SPIRAL is None:
    for cand in _SPIRAL_CANDIDATES:
        p = pathlib.Path(cand)
        if p.exists():
            SPIRAL = p
            break

if SPIRAL is None:
    print("WARNING: units_spiral.json not found. Only phoneme/grapheme/alias seeding will run.")
    print(f"  Tried: {_SPIRAL_CANDIDATES[:4]}")
    SPIRAL = pathlib.Path("/nonexistent")  # will return empty dict

# ── Curated phoneme set (24 English consonants + vowels) ───────────────────
# key = IPA-ish identifier (used as node key: "ipa-f", "ipa-m", …)
# Each entry: {ipa, voice, place, manner, nicknames, example_words}
PHONEMES = {
    "f":  {"ipa": "f",  "voice": False, "place": "Labiodental",   "manner": "Fricative",
           "nicknames": ["Fan Sound", "the fan sound"],
           "examples": ["fan","fish","leaf","coffee"]},
    "v":  {"ipa": "v",  "voice": True,  "place": "Labiodental",   "manner": "Fricative",
           "nicknames": ["Vroom Sound"],
           "examples": ["van","river","love","give"]},
    "m":  {"ipa": "m",  "voice": True,  "place": "Bilabial",      "manner": "Nasal",
           "nicknames": ["Humming Sound", "the humming sound"],
           "examples": ["man","moon","mat","summer"]},
    "n":  {"ipa": "n",  "voice": True,  "place": "Alveolar",      "manner": "Nasal",
           "nicknames": ["Nose Sound", "the nose sound"],
           "examples": ["no","net","can","penny"]},
    "ng": {"ipa": "ŋ",  "voice": True,  "place": "Velar",         "manner": "Nasal",
           "nicknames": ["Ring Sound"],
           "examples": ["sing","ring","finger","long"]},
    "p":  {"ipa": "p",  "voice": False, "place": "Bilabial",      "manner": "Plosive",
           "nicknames": ["Pop Sound"],
           "examples": ["pat","pin","cup","apple"]},
    "b":  {"ipa": "b",  "voice": True,  "place": "Bilabial",      "manner": "Plosive",
           "nicknames": ["Bounce Sound"],
           "examples": ["bat","bin","cab","rubber"]},
    "t":  {"ipa": "t",  "voice": False, "place": "Alveolar",      "manner": "Plosive",
           "nicknames": ["Tongue-Tap Sound"],
           "examples": ["top","cat","sit","letter"]},
    "d":  {"ipa": "d",  "voice": True,  "place": "Alveolar",      "manner": "Plosive",
           "nicknames": ["Drum Sound"],
           "examples": ["dog","dig","bed","ladder"]},
    "k":  {"ipa": "k",  "voice": False, "place": "Velar",         "manner": "Plosive",
           "nicknames": ["Click Sound"],
           "examples": ["cat","kit","duck","packet"]},
    "g":  {"ipa": "g",  "voice": True,  "place": "Velar",         "manner": "Plosive",
           "nicknames": ["Growl Sound"],
           "examples": ["go","got","bag","bigger"]},
    "h":  {"ipa": "h",  "voice": False, "place": "Glottal",       "manner": "Fricative",
           "nicknames": ["Breath Sound"],
           "examples": ["hat","hot","hen","behind"]},
    "l":  {"ipa": "l",  "voice": True,  "place": "Alveolar",      "manner": "Lateral",
           "nicknames": ["Lift Sound"],
           "examples": ["let","log","bell","call"]},
    "r":  {"ipa": "r",  "voice": True,  "place": "Alveolar",      "manner": "Approximant",
           "nicknames": ["Roll Sound"],
           "examples": ["red","ran","car","around"]},
    "w":  {"ipa": "w",  "voice": True,  "place": "Labial-velar",  "manner": "Approximant",
           "nicknames": ["Wind Sound"],
           "examples": ["wet","win","cow","away"]},
    "y":  {"ipa": "j",  "voice": True,  "place": "Palatal",       "manner": "Approximant",
           "nicknames": ["Yell Sound"],
           "examples": ["yes","yam","boy","beyond"]},
    "s":  {"ipa": "s",  "voice": False, "place": "Alveolar",      "manner": "Fricative",
           "nicknames": ["Snake Sound", "the snake sound"],
           "examples": ["sun","sit","bus","class"]},
    "z":  {"ipa": "z",  "voice": True,  "place": "Alveolar",      "manner": "Fricative",
           "nicknames": ["Buzz Sound"],
           "examples": ["zip","zoo","buzz","lazy"]},
    "sh": {"ipa": "ʃ",  "voice": False, "place": "Postalveolar",  "manner": "Fricative",
           "nicknames": ["Quiet Sound"],
           "examples": ["ship","fish","wish","shed"]},
    "zh": {"ipa": "ʒ",  "voice": True,  "place": "Postalveolar",  "manner": "Fricative",
           "nicknames": ["Treasure Sound"],
           "examples": ["measure","vision","beige"]},
    "ch": {"ipa": "tʃ", "voice": False, "place": "Postalveolar",  "manner": "Affricate",
           "nicknames": ["Choo-Choo Sound"],
           "examples": ["chip","chat","rich","watch"]},
    "j":  {"ipa": "dʒ", "voice": True,  "place": "Postalveolar",  "manner": "Affricate",
           "nicknames": ["Jump Sound"],
           "examples": ["jam","jet","judge","age"]},
    "th": {"ipa": "θ",  "voice": False, "place": "Dental",        "manner": "Fricative",
           "nicknames": ["Tongue-Peek Sound"],
           "examples": ["think","thumb","bath","three"]},
    "dh": {"ipa": "ð",  "voice": True,  "place": "Dental",        "manner": "Fricative",
           "nicknames": ["Voice-Tongue-Peek Sound"],
           "examples": ["this","that","mother","father"]},
}

# ── Grapheme → phoneme mapping (common English GPCs) ───────────────────────
# Each grapheme maps to one or more phoneme keys (most common mapping first).
GRAPHEME_MAP: dict[str, list[str]] = {
    "f":  ["f"],
    "ph": ["f"],
    "gh": ["f"],          # rough, enough (rare /f/ spelling)
    "v":  ["v"],
    "m":  ["m"],
    "n":  ["n"],
    "kn": ["n"],          # knee, knife
    "gn": ["n"],          # gnome, sign
    "ng": ["ng"],
    "p":  ["p"],
    "b":  ["b"],
    "t":  ["t"],
    "d":  ["d"],
    "k":  ["k"],
    "c":  ["k", "s"],     # cat (k), city (s)
    "ck": ["k"],
    "ch": ["ch"],
    "tch":["ch"],
    "g":  ["g", "j"],     # go (g), gem (j)
    "dge":["j"],
    "j":  ["j"],
    "h":  ["h"],
    "wh": ["w", "h"],     # when (w), who (h)
    "w":  ["w"],
    "l":  ["l"],
    "r":  ["r"],
    "wr": ["r"],          # write, wrong
    "y":  ["y"],
    "s":  ["s", "z"],     # sun (s), rose (z)
    "z":  ["z"],
    "sh": ["sh"],
    "th": ["th", "dh"],   # thin (th), this (dh)
    "x":  ["ks"],          # fox (not a simple phoneme, but common)
    "qu": ["kw"],          # queen (consonant cluster, not a single phoneme)
}
# Remove any grapheme that maps to a phoneme key not in PHONEMES
GRAPHEME_MAP = {g: [p for p in ps if p in PHONEMES]
                for g, ps in GRAPHEME_MAP.items()}
GRAPHEME_MAP = {g: ps for g, ps in GRAPHEME_MAP.items() if ps}

# ── Root-piece → likely first phoneme (used to attach words to phonemes) ──
ROOT_TO_FIRST_PHONEME: dict[str, str] = {
    "tri": "t", "uni": "y", "bi": "b",
    "octo": "o", "cent": "s", "dent": "d", "ped": "p",
    "manu": "m", "aqua": "a", "sol": "s", "luna": "l",
    "terr": "t", "flor": "f", "zoo": "z", "nov": "n",
    "mari": "m", "astro": "a", "aero": "a", "bio": "b",
    "geo": "j", "micro": "m", "mega": "m", "scope": "s",
    "port": "p", "matr": "m", "patr": "p", "frater":"f",
    "herb": "h", "anim": "a", "nam": "n",
    "spect": "s", "dict": "d", "scrib": "s", "graph": "g",
    "tele": "t", "photo": "f", "phon": "f", "struct":"s",
    "vid": "v", "therm": "th", "mot": "m", "rupt": "r",
    "tract": "t", "multi": "m", "mort": "m",
}


# ── Node creation helpers ───────────────────────────────────────────────────

CHUNK_SEED = -1  # marker: this node/edge is seeded, not from a real document


def _node_key(name: str, ntype: str) -> str:
    """Scoped node key like 'ipa-f', 'prefix-bi', 'meaning-two'."""
    return normalize(f"{ntype}-{name}")


def _ensure_node(kg: KnowledgeGraph, key: str, display: str, ntype: str,
                 desc: str = "", extra: dict | None = None) -> str:
    """Create or update a node with idempotent behaviour."""
    if kg.g.has_node(key):
        node = kg.g.nodes[key]
        # Update display if better available
        if display and len(display) > len(node.get("display", "")):
            node["display"] = display
        if ntype and ntype != "unknown":
            node["type"] = ntype
        if desc and desc not in node.setdefault("descriptions", []):
            node["descriptions"].append(desc)
        if CHUNK_SEED not in node.setdefault("chunk_ids", []):
            node["chunk_ids"].append(CHUNK_SEED)
        if extra:
            for k, v in extra.items():
                node.setdefault(k, v)
    else:
        node_data = {
            "display": display,
            "type": ntype,
            "descriptions": [desc] if desc else [],
            "chunk_ids": [CHUNK_SEED],
        }
        if extra:
            node_data.update(extra)
        kg.g.add_node(key, **node_data)
    return key


def _edge(kg: KnowledgeGraph, src: str, dst: str, etype: str,
          desc: str = "") -> bool:
    """Add a directed edge if it doesn't already exist (same src,dst,type).
    Auto-creates missing endpoints as unknown-typed nodes."""
    s, t = normalize(src), normalize(dst)
    if not s or not t or s == t:
        return False
    # Auto-create endpoints if missing
    for nk, display in ((s, src), (t, dst)):
        if not kg.g.has_node(nk):
            kg.g.add_node(nk, display=display, type="unknown",
                          descriptions=[], chunk_ids=[CHUNK_SEED])
    # Check existing parallel edges of same type
    for _ekey, data in kg.g[s].get(t, {}).items():
        if data.get("type") == etype:
            if desc and desc not in data.setdefault("descriptions", []):
                data["descriptions"].append(desc)
            return False
    kg.g.add_edge(s, t, type=etype, descriptions=[desc] if desc else [],
                   chunk_ids=[CHUNK_SEED])
    return True


# ── Main seeder ─────────────────────────────────────────────────────────────

def seed(kg: KnowledgeGraph) -> dict:
    """Run the full structural-linguistics enrichment in one pass.

    Returns a stats dict for logging.
    """
    stats: dict[str, int] = defaultdict(int)
    spiral_data = json.loads(SPIRAL.read_text()) if SPIRAL.exists() else {}
    spiral_roots = spiral_data.get("roots", {})

    # ─────────────────────────────────────────────────────────────────
    # 1. Prefix nodes — extracted from root pieces of the spiral
    # ─────────────────────────────────────────────────────────────────
    seen_prefixes: set[str] = set()
    for _key, r in spiral_roots.items():
        piece = r["piece"]
        meaning = r["meaning"]
        g = r.get("grade", 1)
        pkey = _node_key(piece.lower(), "prefix")
        _ensure_node(kg, pkey, piece, "prefix",
                     f"Prefix meaning '{meaning}'",
                     {"grade": g, "meaning": meaning})
        seen_prefixes.add(pkey)
        stats["prefix_nodes"] += 1

    # ─────────────────────────────────────────────────────────────────
    # 1b. Combo words (compound roots like telescope = tele + scope)
    # ─────────────────────────────────────────────────────────────────
    combos = spiral_data.get("_combos", [])
    for combo in combos:
        w = combo["w"]
        a_root = combo.get("a", "")
        b_root = combo.get("b", "")
        gloss = combo.get("g", "")
        # Create the combo word as a proper word node
        wkey = normalize(w)
        _ensure_node(kg, wkey, w, "word",
                     f"{gloss} (combines {a_root} + {b_root})",
                     {"is_combo": True})
        # Link combo word to both root parts
        for root_piece in [a_root, b_root]:
            if not root_piece:
                continue
            _edge(kg, normalize(root_piece), wkey, "root_of",
                  f"{w} carries the root {root_piece} combined with another root")
            _edge(kg, normalize(root_piece), wkey, "used_in",
                  f"{root_piece} appears in {w}")
            stats["combo_edges"] += 1
    stats["combo_words"] = len(combos)

    # ─────────────────────────────────────────────────────────────────
    # 2. Meaning nodes — one per unique meaning across all roots
    # ─────────────────────────────────────────────────────────────────
    seen_meanings: set[str] = set()
    for _key, r in spiral_roots.items():
        piece = r["piece"]
        meaning = r["meaning"]
        mkey = _node_key(meaning, "meaning")
        _ensure_node(kg, mkey, meaning, "meaning",
                     f"Meaning: {meaning}")
        _edge(kg, normalize(piece), mkey, "means",
              f"The root {piece} means '{meaning}'")
        seen_meanings.add(mkey)
        stats["meaning_nodes"] += 1

    # ─────────────────────────────────────────────────────────────────
    # 3. Phoneme nodes — 24 curated English phonemes
    # ─────────────────────────────────────────────────────────────────
    for pkey_suffix, data in PHONEMES.items():
        pk = _node_key(pkey_suffix, "phoneme")
        _ensure_node(kg, pk, data["ipa"], "phoneme",
                     f"{'voiced' if data['voice'] else 'voiceless'} "
                     f"{data['place'].lower()} {data['manner'].lower()}",
                     {"voice": data["voice"], "place": data["place"],
                      "manner": data["manner"], "examples": data["examples"]})
        stats["phoneme_nodes"] += 1

    # ─────────────────────────────────────────────────────────────────
    # 4. Grapheme nodes — common English GPCs
    # ─────────────────────────────────────────────────────────────────
    for gph, phoneme_keys in GRAPHEME_MAP.items():
        gk = _node_key(gph, "grapheme")
        _ensure_node(kg, gk, gph, "grapheme",
                     f"Grapheme '{gph}'")
        for pk_suffix in phoneme_keys:
            pk = _node_key(pk_suffix, "phoneme")
            _edge(kg, pk, gk, "written_as",
                  f"The sound {PHONEMES[pk_suffix]['ipa']} is often written as '{gph}'")
            _edge(kg, gk, pk, "sounds_like",
                  f"The letters '{gph}' sound like {PHONEMES[pk_suffix]['ipa']}")
            stats["grapheme_edges"] += 1
        stats["grapheme_nodes"] += 1

    # ─────────────────────────────────────────────────────────────────
    # 5. Alias nodes — sensory nicknames that resolve to phonemes
    # ─────────────────────────────────────────────────────────────────
    for pkey_suffix, data in PHONEMES.items():
        pk = _node_key(pkey_suffix, "phoneme")
        for nickname in data.get("nicknames", []):
            ak = _node_key(nickname, "alias")
            _ensure_node(kg, ak, nickname, "alias",
                         f"Learner-friendly name for the {data['ipa']} sound")
            _edge(kg, ak, pk, "alias_of",
                  f"'{nickname}' refers to the {data['ipa']} phoneme")
            stats["alias_nodes"] += 1

    # ─────────────────────────────────────────────────────────────────
    # 6. Reverse edges (root → used_in → word)
    # ─────────────────────────────────────────────────────────────────
    for u, v, data in list(kg.g.edges(data=True)):
        if data.get("type") == "root_of":
            root_key, word_key = u, v
            root_display = kg.g.nodes.get(root_key, {}).get("display", root_key)
            word_display = kg.g.nodes.get(word_key, {}).get("display", word_key)
            _edge(kg, root_key, word_key, "used_in",
                  f"{root_display} appears in {word_display}")
            stats["reverse_edges"] += 1

    # ─────────────────────────────────────────────────────────────────
    # 7. Discovery edges (prerequisite_for): family → new
    # ─────────────────────────────────────────────────────────────────
    for _key, r in spiral_roots.items():
        piece = r["piece"]
        family_words = [w["w"] for w in r.get("family", [])]
        new_words = [w["w"] for w in r.get("new", [])]
        g = r.get("grade", 1)
        for fw in family_words:
            for nw in new_words:
                _edge(kg, normalize(fw), normalize(nw), "prerequisite_for",
                      f"A child who knows '{fw}' is ready to discover '{nw}' (grade {g})")
                stats["discovery_edges"] += 1

    # ─────────────────────────────────────────────────────────────────
    # 8. Word → phoneme edges (sounds_like from first-phoneme guess)
    # ─────────────────────────────────────────────────────────────────
    for _key, r in spiral_roots.items():
        piece = r["piece"]
        first_phoneme = ROOT_TO_FIRST_PHONEME.get(piece.lower())
        if not first_phoneme:
            continue
        pk = _node_key(first_phoneme, "phoneme")
        pk_display = PHONEMES.get(first_phoneme, {}).get("ipa", first_phoneme)
        all_words = [w["w"] for w in r.get("family", []) + r.get("new", [])]
        for w in all_words:
            _edge(kg, normalize(w), pk, "sounds_like",
                  f"The word '{w}' begins with the {pk_display} sound")
            stats["word_phoneme_edges"] += 1

    # ─────────────────────────────────────────────────────────────────
    # 9. Grade metadata — attach to all spiral-sourced nodes
    # ─────────────────────────────────────────────────────────────────
    for _key, r in spiral_roots.items():
        piece = r["piece"]
        g = r.get("grade", 1)
        for node_key_hint in [normalize(piece)] + \
                [normalize(w["w"]) for w in r.get("family", []) + r.get("new", [])]:
            if kg.g.has_node(node_key_hint):
                node = kg.g.nodes[node_key_hint]
                if "grade" not in node:
                    node["grade"] = g
                    stats["grade_attached"] += 1

    # ─────────────────────────────────────────────────────────────────
    # 10. Cognate edges (cross-lingual: English ↔ Sanskrit/Hindi/Bangla)
    # ─────────────────────────────────────────────────────────────────
    for _key, r in spiral_roots.items():
        conn = r.get("conn", {})
        if not conn:
            continue
        piece = r["piece"]
        root_key = normalize(piece)
        # Create language anchor nodes
        for lang, word in [("Hindi", conn.get("hi", "")),
                           ("Bengali", conn.get("bn", "")),
                           ("Sanskrit", conn.get("src", "")),
                           ("English", piece)]:
            if not word:
                continue
            lk = _node_key(lang.lower(), "language")
            _ensure_node(kg, lk, lang, "language",
                         f"{lang} language")
            # Link the root to its language-coded counterpart
            wlk = _node_key(word, "language-word")
            _ensure_node(kg, wlk, word, "language-word",
                         f"{lang}: {word}")
            _edge(kg, root_key, wlk, "cognate_with" if lang == "Sanskrit"
                                          else "translates_to",
                  f"English root {piece} ↔ {lang} {word}")
            stats["language_edges"] += 1

    # ─────────────────────────────────────────────────────────────────
    # 11. Build discovery communities ─────────────────────────────────
    # ─────────────────────────────────────────────────────────────────
    existing = load_communities(COMMUNITIES_PATH) if COMMUNITIES_PATH.exists() else []
    next_cid = (max((c.id for c in existing), default=-1)) + 1
    new_comms: list[Community] = []

    # Phoneme families
    for psuffix, data in PHONEMES.items():
        pk = _node_key(psuffix, "phoneme")
        member_keys = [pk]
        # Add grapheme nodes that map to this phoneme
        for gph, pks in GRAPHEME_MAP.items():
            if psuffix in pks:
                gk = _node_key(gph, "grapheme")
                member_keys.append(gk)
        # Add alias nodes
        for nickname in data.get("nicknames", []):
            ak = _node_key(nickname, "alias")
            member_keys.append(ak)
        member_keys = [k for k in member_keys if kg.g.has_node(k)]
        if len(member_keys) >= 2:
            title = f"Sound: {data['nicknames'][0]} (/ipa-{psuffix}/)"
            new_comms.append(Community(
                id=next_cid, nodes=member_keys, title=title,
                summary=f"{'Voiced' if data['voice'] else 'Voiceless'} {data['place']} {data['manner']}.",
                findings=[f"Written as: {', '.join(g for g, pks in GRAPHEME_MAP.items() if psuffix in pks)}"],
                size=len(member_keys),
            ))
            next_cid += 1
            stats["phoneme_communities"] += 1

    # Grapheme-pattern communities (digra phs: ph, sh, ch, th, ng)
    digraph_communities = {
        "ph": "The Fan-Sound-in-Disguise /ph/",
        "sh": "The Quiet Sound /sh/",
        "ch": "The Choo-Choo Sound /ch/",
        "th": "The Tongue-Peek Sound /th/",
        "ng": "The Ring Sound /ng/",
    }
    for gph, title in digraph_communities.items():
        gk = _node_key(gph, "grapheme")
        if not kg.g.has_node(gk):
            continue
        member_keys = [gk]
        # Find phoneme keys connected to this grapheme
        for _, dst, data in kg.g.edges(gk, data=True):
            if data.get("type") == "sounds_like" and kg.g.has_node(dst):
                member_keys.append(dst)
        if len(member_keys) >= 2:
            new_comms.append(Community(
                id=next_cid, nodes=member_keys, title=title,
                summary=f"The letters '{gph}' represent a single sound. "
                        f"This is called a digraph — two letters making one sound.",
                findings=[f"Practice words: {', '.join(PHONEMES.get(gph, {}).get('examples', []))}"],
                size=len(member_keys),
            ))
            next_cid += 1
            stats["grapheme_communities"] += 1

    save_communities(existing + new_comms, COMMUNITIES_PATH)
    stats["new_communities"] = len(new_comms)

    return dict(stats)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    kg = KnowledgeGraph.load(GRAPH_PATH)
    before_nodes = kg.g.number_of_nodes()
    before_edges = kg.g.number_of_edges()
    before_comms = (len(load_communities(COMMUNITIES_PATH))
                    if COMMUNITIES_PATH.exists() else 0)

    stats = seed(kg)

    kg.save(GRAPH_PATH)
    after_nodes = kg.g.number_of_nodes()
    after_edges = kg.g.number_of_edges()
    after_comms = (len(load_communities(COMMUNITIES_PATH))
                   if COMMUNITIES_PATH.exists() else 0)

    print(f"Graph: {before_nodes}→{after_nodes} nodes (+{after_nodes-before_nodes})")
    print(f"Edges:  {before_edges}→{after_edges} edges (+{after_edges-before_edges})")
    print(f"Communities: {before_comms}→{after_comms} (+{after_comms-before_comms})")
    print()
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print()
    print("✓ Phase 1 structural linguistics graph seeded.")
    print("  Restart the backend to load the enriched graph.")
