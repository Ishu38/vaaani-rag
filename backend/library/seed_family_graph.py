#!/usr/bin/env python3
"""Seed person/family word-family themes into the knowledge graph.

Creates entities for person(s), mother, father, brother, sister, parent,
child, family, baby, friend, neighbor, grandfather, grandmother, uncle,
aunt, cousin, and relations between them — all connected to word-formation
patterns so the camera discovery can ground "person" and family objects in
the graph.

Idempotent: re-running skips entities that already exist.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from any directory — resolve relative to this file's location
HERE = Path(__file__).resolve().parent        # backend/library/
REPO = HERE.parent                              # backend/
ROOT = REPO.parent                              # rag-assistant/
sys.path.insert(0, str(REPO))

from graph import KnowledgeGraph, normalize
from community import Community


GRAPH_PATH = ROOT / "data" / "graph.json"
COMMUNITIES_PATH = ROOT / "data" / "communities.json"

# ── Person/Family word families ────────────────────────────────────────
# Each root concept links to words formed from it. The pattern is:
#   root entity (type: root/pattern) → word entities (type: word)
#   with a "word_family" or "derived_from" relation.
# This mirrors the existing TRI→triangle, SOL→solar pattern.

FAMILY_DATA = [
    # (root, root_type, words_derived, community_title, community_story)
    # NOTE: these are THEMATIC clusters (type:"theme"), NOT morphological
    # root families.  Real Latin/Greek morphological roots (BI, TRI, AQUA,
    # SOL, PORT, DICT…) are seeded by build_library_graph.py from the
    # vetted spiral curriculum and use type:"root".  Thematic clusters
    # exist so the discovery system has a graph to attach discoveries to
    # (COCO-SSD detects "chair", "dog", "person" — these nodes give those
    # objects a home).  They must never masquerade as morphological roots
    # or they pollute the Root Bridge pedagogy.
    (
        "PERSON", "theme",
        ["person", "persons", "people", "personal", "personality", "personify"],
        "Theme: PERSON — One, Many, and Beyond",
        "The Latin root PERSONA means a mask worn by an actor, and from it we get person (one human), persons/people (many humans), personal (belonging to one person), personality (their unique character), and personify (to give human traits to something).",
    ),
    (
        "FATHER", "theme",
        ["father", "fatherly", "fatherhood", "grandfather", "dad", "daddy"],
        "Theme: FATHER — The Protector Names",
        "English 'father' comes from Old English fæder, related to Latin pater. Family words build outward: father is one parent, grandfather is two generations up, fatherly describes care like a dad gives, and fatherhood is the state of being a dad.",
    ),
    (
        "MOTHER", "theme",
        ["mother", "motherly", "motherhood", "grandmother", "mom", "mommy"],
        "Theme: MOTHER — Nurture Words",
        "Old English modor connects to Latin mater. The family tree: mother (one parent), grandmother (two generations up), motherly (gentle and caring), and motherhood (being a mother). Every culture has a special mother-word.",
    ),
    (
        "PARENT", "theme",
        ["parent", "parents", "parental", "parenthood", "parenting", "guardian"],
        "Theme: PARENT — Caregivers",
        "From Latin parens (one who brings forth). A parent protects and raises a child. Parents is the plural. Parental love is a parent's care. Parenthood is the job of being a parent. Guardian is a person who looks after someone.",
    ),
    (
        "CHILD", "theme",
        ["child", "children", "childhood", "childlike", "childish", "baby", "infant", "toddler"],
        "Theme: CHILD — Growing Up",
        "Old English cild means a young human. Children is the irregular plural. Childhood is your growing-up years. Childlike means innocent wonder (positive). Childish means immature (negative). Baby and infant are the youngest children. Toddler is a child learning to walk.",
    ),
    (
        "FAMILY", "theme",
        ["family", "families", "familiar", "familial", "relative", "kin", "household"],
        "Theme: FAMILY — Bonds and Belonging",
        "Latin familia means a household including servants. Family ties extend through blood and love. Families is the plural. Familiar means well-known (like family). Kin are blood relatives. A household is everyone living under one roof.",
    ),
    (
        "BROTHER", "theme",
        ["brother", "brothers", "brotherly", "brotherhood", "sibling", "stepbrother"],
        "Theme: BROTHER — Brotherhood Bonds",
        "Old English brothor, related to Latin frater. A brother is a male sibling. Brotherly means friendly and loyal (like brothers are). Brotherhood is the bond between brothers — or any close group. A sibling is any brother or sister.",
    ),
    (
        "SISTER", "theme",
        ["sister", "sisters", "sisterly", "sisterhood", "sibling", "stepsister"],
        "Theme: SISTER — Sisterhood",
        "Old English sweostor, related to Latin soror. A sister is a female sibling. Sisterly means kind and protective. Sisterhood is the close bond between sisters. The word sibling covers both brothers and sisters.",
    ),
    (
        "FRIEND", "theme",
        ["friend", "friends", "friendly", "friendship", "befriend", "unfriendly", "friendless"],
        "Theme: FRIEND — Companions",
        "Old English freond means one who loves. A friend is someone you care about. Friends are many. Friendly describes a warm nature. Friendship is the bond. Befriend means make friends. Unfriendly means cold or mean. Friendless means having no friends — a sad word.",
    ),
    (
        "NEIGHBOR", "theme",
        ["neighbor", "neighbours", "neighbourhood", "neighbourly", "neighbouring", "neighborly"],
        "Theme: NEIGHBOR — Near and Next-Door",
        "Old English neahgebur — 'near' + 'dweller'. A neighbor lives next to you. Neighbourhood is the area where neighbours live. Neighbourly means helpful (like a good neighbour). Neighbouring means next to something.",
    ),
    (
        "MAN", "theme",
        ["man", "men", "manly", "manhood", "mankind", "gentleman", "human"],
        "Theme: MAN — The Human Root",
        "Old English mann meant human (not just male). Man = one adult male, men = many. Mankind = all humans. Human comes from Latin humanus, sharing the same ancient root. Gentleman is a polite, honorable man.",
    ),
    (
        "WOMAN", "theme",
        ["woman", "women", "womanly", "womanhood", "womankind", "lady", "female"],
        "Theme: WOMAN — Womanhood",
        "Old English wifman — 'wife' + 'human'. Woman = one adult female, women = many. Womanly means graceful and strong. Womanhood is being a woman. Lady is a polite word for woman. Female is the biological word.",
    ),
    (
        "BOY / GIRL", "theme",
        ["boy", "boys", "boyhood", "boyish", "girl", "girls", "girlhood", "girlish"],
        "Theme: BOY and GIRL — Young Ones",
        "Boy and girl are the words for young humans. A boy is a male child; a girl is a female child. Boyhood and girlhood are your growing years. Boyish means like a boy (playful, rough). Girlish means like a girl (sweet, lively). These words teach us that language marks age and gender differently.",
    ),
    # ── Everyday objects (COCO-SSD detectable) ──────────────────────
    (
        "CHAIR", "theme",
        ["chair", "chairs", "armchair", "wheelchair", "bench", "seat", "throne", "stool"],
        "Theme: CHAIR — Places to Sit",
        "Old French chaiere, from Latin cathedra (seat). A chair has a back and four legs. An armchair has armrests. A wheelchair helps people move. A bench is a long chair for many people. A stool has no back. A throne is a special chair for a king or queen.",
    ),
    (
        "TABLE", "theme",
        ["table", "tables", "tabletop", "tablecloth", "timetable", "tablet", "dining table"],
        "Theme: TABLE — Flat Surfaces",
        "Latin tabula means a flat board. A table is a flat top with legs. A tablecloth covers it for meals. A dining table is for eating together. A timetable is a chart of times. A tablet was once a flat writing surface.",
    ),
    (
        "CUP", "theme",
        ["cup", "cups", "teacup", "cupful", "cupcake", "cupboard"],
        "Theme: CUP — Drinking Vessels",
        "Old English cuppe from Latin cupa (tub). A cup holds liquid for drinking. A teacup is a small cup for tea. A cupboard is a cabinet where cups live (board = shelf). A cupcake is a tiny cake baked in a cup-shaped mold.",
    ),
    (
        "BOTTLE", "theme",
        ["bottle", "bottles", "bottled", "bottlecap", "water bottle", "feeding bottle"],
        "Theme: BOTTLE — Liquid Containers",
        "Old French bouteille from Latin buttis (cask). A bottle holds liquids. A water bottle is for carrying water. A feeding bottle feeds a baby. Bottled means sealed in bottles. A bottlecap keeps the liquid inside.",
    ),
    (
        "BOOK", "theme",
        ["book", "books", "booklet", "bookcase", "bookmark", "notebook", "storybook", "textbook"],
        "Theme: BOOK — Written Worlds",
        "Old English boc means a written document. A book holds stories and knowledge. A booklet is a small book. A bookcase stores books. A bookmark saves your place. A notebook is for writing. A storybook tells tales. A textbook teaches.",
    ),
    (
        "DOG", "theme",
        ["dog", "dogs", "doghouse", "doggy", "puppy", "hound", "canine"],
        "Theme: DOG — Our Oldest Friends",
        "Old English docga. A dog is a four-legged friend. Dogs is the plural. A puppy is a baby dog. A doghouse is a tiny house for a dog. Doggy is a cute name. Hound is a hunting dog. Canine means related to dogs.",
    ),
    (
        "CAT", "theme",
        ["cat", "cats", "catnip", "catlike", "caterwaul", "kitten", "feline"],
        "Theme: CAT — Independent Companions",
        "Old English catt from Latin catta. A cat is a small furry pet. A kitten is a baby cat. Catlike means graceful and quiet. Feline means related to cats. Catnip is a plant cats love. Caterwaul is a loud cat cry.",
    ),
    (
        "FOOD / EAT", "theme",
        ["food", "foods", "feed", "foodie", "apple", "banana", "bread", "pizza", "rice", "sandwich", "cake", "carrot"],
        "Theme: FOOD — What We Eat",
        "Old English foda. Food gives us energy. To feed means give food. An apple is a round fruit. A banana is a yellow fruit. Bread is made from flour. Pizza comes from Italy. Carrots are orange vegetables. Cake is a sweet treat for special days.",
    ),
    # VEHICLE entry removed — it was a thematic cluster masquerading as a
    # morphological root and collided with the real curriculum root BI →
    # bicycle (seeded by build_library_graph.py from units_spiral.json).
    # The BI family (bi, bicycle, binoculars, biped, bilingual, biweekly)
    # is the correct 6-node community for "bicycle".
    (
        "HOME", "theme",
        ["home", "house", "homes", "homely", "homework", "homeless", "bed", "door", "window", "roof", "kitchen", "bathroom", "garden"],
        "Theme: HOME — Where We Live",
        "Old English ham means dwelling. A home is where you live. A house is the building. Homely means cosy and welcoming. Homework is schoolwork done at home. A bed is for sleeping. A door lets you in. A window lets in light. A roof keeps rain out. A kitchen is for cooking.",
    ),
]

# ── Relations: every derived word links back to its root ──────────────
# Also cross-link related roots (FATHER → PARENT, BROTHER → SISTER, etc.)
CROSS_LINKS = [
    ("father", "parent", "is_a", "A father is a kind of parent."),
    ("mother", "parent", "is_a", "A mother is a kind of parent."),
    ("father", "mother", "partner", "Father and mother are partners in parenting."),
    ("brother", "sister", "sibling", "A brother and a sister are siblings."),
    ("brother", "family", "member_of", "A brother is a family member."),
    ("sister", "family", "member_of", "A sister is a family member."),
    ("parent", "family", "member_of", "A parent is a family member."),
    ("child", "family", "member_of", "A child is a family member."),
    ("grandfather", "father", "parent_of", "A grandfather is the father of the father or mother."),
    ("grandmother", "mother", "parent_of", "A grandmother is the mother of the father or mother."),
    ("friend", "neighbor", "related", "A friend may be a neighbour, but a neighbour is not always a friend."),
    ("man", "person", "is_a", "A man is a person."),
    ("woman", "person", "is_a", "A woman is a person."),
    ("boy", "child", "is_a", "A boy is a child."),
    ("girl", "child", "is_a", "A girl is a child."),
    ("human", "person", "synonym", "'Human' and 'person' mean roughly the same."),
    ("people", "persons", "synonym", "'People' is the everyday plural of person; 'persons' is formal."),
    ("person", "family", "part_of", "Every person belongs to a family."),
    # Everyday object cross-links
    ("chair", "home", "found_in", "Chairs are found in homes."),
    ("table", "home", "found_in", "Tables are found in homes."),
    ("cup", "home", "found_in", "Cups are found in homes."),
    ("bed", "home", "found_in", "Beds are found in homes — and every home needs one."),
    ("book", "home", "found_in", "Books are often found in homes."),
    ("chair", "table", "paired_with", "Chairs are often found next to tables — they're the perfect pair."),
    ("cup", "table", "placed_on", "A cup is often placed on a table."),
    ("dog", "home", "lives_in", "A dog often lives in a home as a pet."),
    ("cat", "home", "lives_in", "A cat often lives in a home as a pet."),
    ("dog", "cat", "companion", "Dogs and cats are two of the most common pets — they are companions to people."),
    ("apple", "food", "is_a", "An apple is a kind of food — a fruit."),
    ("banana", "food", "is_a", "A banana is a kind of food — a fruit."),
    ("carrot", "food", "is_a", "A carrot is a kind of food — a vegetable."),
    ("pizza", "food", "is_a", "Pizza is a kind of food from Italy."),
    ("book", "cup", "paired_with", "Many people enjoy reading a book with a cup of tea — a cosy pair."),
]


def add_entity(g: KnowledgeGraph, name: str, etype: str, desc: str = ""):
    """Add a single entity with a synthetic chunk_id=-1 (seeded)."""
    key = normalize(name)
    if g.g.has_node(key):
        return key  # already present — idempotent
    g.g.add_node(
        key,
        display=name,
        type=etype,
        descriptions=[desc] if desc else [],
        chunk_ids=[-1],  # marker: seeded, not from real document
    )
    return key


def add_relation(g: KnowledgeGraph, src: str, dst: str, rtype: str, desc: str = ""):
    """Add a relationship between two entities."""
    s, t = normalize(src), normalize(dst)
    if not s or not t or s == t:
        return False
    # Auto-create if missing
    for k, raw in ((s, src), (t, dst)):
        if not g.g.has_node(k):
            g.g.add_node(k, display=raw, type="word", descriptions=[], chunk_ids=[-1])
    # Check for existing parallel edge of same type
    for _k, data in g.g[s].get(t, {}).items():
        if data.get("type") == rtype:
            return False  # already exists
    g.g.add_edge(
        s, t,
        type=rtype,
        descriptions=[desc] if desc else [],
        chunk_ids=[-1],
    )
    return True


def main():
    print("Loading existing knowledge graph...")
    kg = KnowledgeGraph()
    if GRAPH_PATH.exists():
        from networkx.readwrite import node_link_data, node_link_graph
        with open(GRAPH_PATH) as f:
            raw = json.load(f)
        kg.g = node_link_graph(raw, edges="links")

    before_nodes = kg.g.number_of_nodes()
    before_edges = kg.g.number_of_edges()

    added_entities = 0
    added_relations = 0

    for root, rtype, words, _title, _story in FAMILY_DATA:
        # Add root entity (thematic cluster, NOT a morphological root)
        key = add_entity(kg, root, rtype, f"Thematic cluster: {root}")
        if not kg.g.nodes[key].get("descriptions"):
            kg.g.nodes[key]["descriptions"] = [f"Thematic cluster: {root}"]
        else:
            added_entities += 1

        # Add derived words
        for w in words:
            wk = add_entity(kg, w, "word", f"A word related to the theme {root}")
            if not kg.g.nodes[wk].get("descriptions"):
                kg.g.nodes[wk]["descriptions"] = [f"A word derived from or related to the root {root}"]
            else:
                # already existed, check relation
                pass
            # Link word to root
            if add_relation(kg, root, w, "word_family", f"{w} belongs to the {root} word family."):
                added_relations += 1

    # Cross-links between roots
    for src, dst, rtype, desc in CROSS_LINKS:
        if add_relation(kg, src, dst, rtype, desc):
            added_relations += 1

    after_nodes = kg.g.number_of_nodes()
    after_edges = kg.g.number_of_edges()

    from networkx.readwrite import node_link_data
    data = node_link_data(kg.g, edges="links")
    with open(GRAPH_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Graph: {before_nodes}→{after_nodes} nodes, {before_edges}→{after_edges} edges")
    print(f"  +{after_nodes - before_nodes} entities, +{after_edges - before_edges} relations")
    print(f"Saved → {GRAPH_PATH}")

    # ── Build communities ──────────────────────────────────────────
    communities: list[Community] = []

    if COMMUNITIES_PATH.exists():
        with open(COMMUNITIES_PATH) as f:
            raw_coms = json.load(f)
        for c in raw_coms:
            communities.append(Community.from_dict(c))
        next_id = max(c.id for c in communities) + 1
    else:
        next_id = 0

    new_communities = 0
    for root, _rtype, words, title, story in FAMILY_DATA:
        # Check if this community already exists (by title match)
        if any(c.title == title for c in communities):
            continue

        node_keys = [normalize(k) for k in ([root] + words) if kg.g.has_node(normalize(k))]
        if not node_keys:
            continue

        findings = []
        for w in words[:6]:
            findings.append(f"{w} is a word in the {root} family.")

        communities.append(Community(
            id=next_id,
            nodes=node_keys,
            title=title,
            summary=story,
            findings=findings,
            size=len(node_keys),
        ))
        next_id += 1
        new_communities += 1

    communities.sort(key=lambda c: c.id)
    with open(COMMUNITIES_PATH, "w") as f:
        json.dump([c.to_dict() for c in communities], f, indent=2)

    print(f"Communities: {len(communities)} total, +{new_communities} new")
    print(f"Saved → {COMMUNITIES_PATH}")
    print("\n✓ Done. Person/family themes seeded into the knowledge graph.")


if __name__ == "__main__":
    main()
