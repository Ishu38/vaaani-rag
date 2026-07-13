"""Error classifier — deterministic, no LLM/SLM.

Classifies student errors into cognitive patterns using rule-based heuristics.
Every rule is traceable, explainable, and instant.
"""

import hashlib
from dataclasses import dataclass, field

ERROR_TYPES = [
    "memorization_override",
    "conceptual_gap",
    "terminology_confusion",
    "spelling_sound_conflation",
    "l1_transfer",
    "overgeneralisation",
    "overconfidence",
    "underconfidence",
    "impulsive",
    "shortcut_dependency",
    "fragile_understanding",
    "visualization_weakness",
    "no_error",
    "sign_error",
    "dimensional_error",
]

# ── Known L1 transfer patterns (Indian English) ────────────────────

_L1_PATTERNS = [
    # Grammar patterns characteristic of Hindi/Bengali transfer
    ("myself", r"\bmyself\b"),                # "Myself Rahul" instead of "I am Rahul"
    ("good_name", r"\bgood\s+name\b"),        # "What is your good name?"
    ("discuss_about", r"\bdiscuss(?:ing|ed)?\s+about\b"),  # "discuss about"
    ("having_own", r"\bis\s+having\b"),       # "is having a car" for stative possession
    ("knowing", r"\bam\s+knowing\b|\bis\s+knowing\b"),  # "am knowing" for stative
    ("works_uncountable", r"\bmuch\s+works\b"),  # "much works"
    ("home_language_order", r"\b(?:very|too|so)\s+\w+\s+\w+\s+hot\b"),  # adj order
    ("present_perfect_past", r"\bhave\s+\w+ed\s+yesterday\b"),  # perfect with past time
    ("missing_article", r"^[A-Z][a-z]+\s+(?:is|was|are|were)"),  # "Dog is" instead of "The dog is"
    ("redundant_preposition", r"\b(?:enter|reach|return|marry|discuss)\w*\s+(?:in|into|to|at|about)"),
    ("subject_verb_number", r"\b(?:he|she|it|[A-Z][a-z]+)\s+(?:do|have|go|run|make|take)\b"),  # "He go"
]

# ── Spelling-sound conflation patterns ─────────────────────────────

_SOUND_SPELL_CONFUSIONS = [
    ({"f", "ph"}, "The /f/ sound can be spelled 'f' (fan) or 'ph' (phone)."),
    ({"k", "c", "ck", "ch"}, "The /k/ sound can be spelled 'c' (cat), 'k' (kite), 'ck' (back), or 'ch' (school)."),
    ({"j", "g", "dge"}, "The /dʒ/ sound can be 'j' (jam), 'g' (gem), or 'dge' (bridge)."),
    ({"s", "c"}, "The /s/ sound can be 's' (sun) or 'c' (cent)."),
    ({"z", "s"}, "The /z/ sound can be 'z' (zoo) or 's' (rose)."),
    ({"ee", "ea", "ie", "ei"}, "The /iː/ sound has multiple spellings: 'ee' (see), 'ea' (sea), 'ie' (piece)."),
    ({"oo", "u", "ou"}, "The /ʊ/ sound can be spelled 'oo' (book), 'u' (put), or 'ou' (could)."),
]

# ── Terminology confusion pairs ────────────────────────────────────

_TERM_CONFUSIONS = [
    ({"letter", "sound", "phoneme"}, "Don't confuse the letter (what you write) with the sound/phoneme (what you say)."),
    ({"vowel", "consonant"}, "A vowel is a sound made with an open vocal tract; a consonant has obstruction."),
    ({"phonetics", "phonology"}, "Phonetics = physical sounds; Phonology = sound systems and patterns."),
    ({"root", "prefix", "suffix"}, "A root carries meaning; prefix attaches before, suffix after."),
    ({"morpheme", "syllable"}, "A morpheme is the smallest unit of meaning; a syllable is a unit of pronunciation."),
    ({"noun", "verb", "adjective"}, "Check the word class — is it naming (noun), doing (verb), or describing (adjective)?"),
]


@dataclass
class ErrorDiagnosis:
    primary_error: str = "no_error"
    explanation: str = ""
    root_cause_topic: str = ""
    remediation: str = ""
    confidence_calibration: str = "well_calibrated"
    error_signature: str = ""
    is_valid: bool = False


def _signature_hash(sig: str) -> str:
    return hashlib.sha1(sig.encode()).hexdigest()[:12]


# ── Deterministic classification rules ─────────────────────────────

def classify_error(
    query: str,
    student_answer: str,
    correct_answer: str,
    topic: str = "",
    confidence_1to5: int = 0,
    response_ms: float = 0,
    is_correct: bool = False,
) -> ErrorDiagnosis:
    """Deterministic error classification — zero LLM calls.

    Applies rule-based heuristics in priority order. Returns a
    complete ErrorDiagnosis with explanation and remediation.
    """

    # ── Step 0: Run the fast coarse check first ──────────────────
    coarse = quick_coarse_check(student_answer, correct_answer,
                                response_ms, confidence_1to5)
    if coarse.is_valid:
        return coarse

    sa = student_answer.strip().lower()
    ca = correct_answer.strip().lower()

    # ── Overconfidence / underconfidence ─────────────────────────
    if confidence_1to5 >= 4 and not is_correct:
        return ErrorDiagnosis(
            primary_error="overconfidence",
            explanation=f"High confidence ({confidence_1to5}/5) but answer is wrong — "
                         f"the student is not aware they don't know this.",
            root_cause_topic=topic or "metacognition",
            remediation="Pause before answering — ask yourself 'Can I explain WHY this is right?' "
                        "before you submit.",
            error_signature="overconfidence::high_confidence_wrong",
            is_valid=True,
        )
    if confidence_1to5 <= 2 and is_correct:
        return ErrorDiagnosis(
            primary_error="underconfidence",
            explanation=f"Low confidence ({confidence_1to5}/5) but answer is correct — "
                         f"the student knows more than they think.",
            root_cause_topic=topic or "self_efficacy",
            remediation="You got it right! Trust what you know — your first instinct was correct.",
            error_signature="underconfidence::low_confidence_right",
            is_valid=True,
        )

    # ── Spelling-sound conflation ─────────────────────────────────
    if _word_diff_by_sound_spelling(sa, ca):
        return ErrorDiagnosis(
            primary_error="spelling_sound_conflation",
            explanation=f"The student is reasoning from spelling (letters) rather than "
                         f"sounds (phonology). The sound is the same but the spelling differs.",
            root_cause_topic="phoneme_grapheme_correspondence",
            remediation="Say the word aloud and listen — don't look at the letters. "
                        "What sound do you actually hear?",
            error_signature="spelling_sound_conflation::letter_bias",
            is_valid=True,
        )

    # ── L1 transfer ─────────────────────────────────────────────
    l1 = _detect_l1_transfer(sa, query)
    if l1:
        return ErrorDiagnosis(
            primary_error="l1_transfer",
            explanation=l1,
            root_cause_topic="l1_interference",
            remediation="Your home language structures the sentence differently — "
                       "that's normal. Compare: how would you say this in your mother tongue? "
                       "Now notice how English does it differently.",
            error_signature=f"l1_transfer::{l1.split(':')[0] if ':' in l1 else 'grammar'}",
            is_valid=True,
        )

    # ── Terminology confusion ────────────────────────────────────
    term = _detect_terminology_confusion(sa, ca, query, topic)
    if term:
        return ErrorDiagnosis(
            primary_error="terminology_confusion",
            explanation=term[0],
            root_cause_topic=term[1],
            remediation=term[2],
            error_signature=f"terminology_confusion::{_signature_hash(term[1])}",
            is_valid=True,
        )

    # ── Overgeneralisation ───────────────────────────────────────
    over = _detect_overgeneralisation(sa, ca)
    if over:
        return ErrorDiagnosis(
            primary_error="overgeneralisation",
            explanation=over,
            root_cause_topic=topic or "grammar_rule_application",
            remediation="You applied a rule that works most of the time, but this is an "
                       "exception. Learn the exceptions alongside the rules — can you think of "
                       "another word that breaks this rule?",
            error_signature="overgeneralisation::rule_overapplication",
            is_valid=True,
        )

    # ── Shortcut dependency ──────────────────────────────────────
    if _detect_shortcut(sa, query):
        return ErrorDiagnosis(
            primary_error="shortcut_dependency",
            explanation="The answer uses a surface trick (pattern matching, keyword spotting) "
                         "without understanding the underlying concept.",
            root_cause_topic=topic or "deep_understanding",
            remediation="Don't just spot keywords — understand WHY. "
                       "Explain the concept in your own words before answering.",
            error_signature="shortcut_dependency::surface_match",
            is_valid=True,
        )

    # ── Fragile understanding ────────────────────────────────────
    if is_correct and (response_ms > 20000 or (confidence_1to5 and confidence_1to5 <= 3)):
        return ErrorDiagnosis(
            primary_error="fragile_understanding",
            explanation=f"Correct but {'took >20s' if response_ms > 20000 else 'low confidence'} "
                         f"— the knowledge is there but not solid yet.",
            root_cause_topic=topic or "automaticity",
            remediation="Practice this until it becomes automatic. "
                       "Try answering the same kind of question 3 more times — speed and confidence will grow.",
            error_signature="fragile_understanding::correct_but_fragile",
            is_valid=True,
        )

    # ── Default: conceptual gap ──────────────────────────────────
    return ErrorDiagnosis(
        primary_error="conceptual_gap",
        explanation="The student's answer suggests a gap in understanding the core concept.",
        root_cause_topic=topic or "fundamental_concept",
        remediation="Let's step back. Before we try this question, what do you already "
                   "know about this topic? Build from what you understand.",
        error_signature="conceptual_gap::broad_misunderstanding",
        is_valid=True,
    )


# ── Heuristic detectors ────────────────────────────────────────────

def _word_diff_by_sound_spelling(sa: str, ca: str) -> bool:
    """True if the answers differ by a known sound-spelling alternation."""
    for spell_set, _ in _SOUND_SPELL_CONFUSIONS:
        sa_has = any(p in sa for p in spell_set)
        ca_has = any(p in ca for p in spell_set)
        if sa_has and ca_has:
            # Check if replacing the spelling makes them match
            for p in spell_set:
                for q in spell_set:
                    if p != q and sa.replace(p, q) == ca:
                        return True
    return False


def _detect_l1_transfer(sa: str, query: str) -> str:
    """Return an explanation if an L1 transfer pattern is detected."""
    import re
    for name, pattern in _L1_PATTERNS:
        if re.search(pattern, sa, re.IGNORECASE):
            explanations = {
                "myself": "Using 'Myself X' instead of 'I am X' — this is a direct "
                          "translation of Hindi 'Mera naam X hai' structure.",
                "good_name": "'Good name' is a calque from Hindi/Bengali 'shubh naam' — "
                             "English just uses 'name'.",
                "discuss_about": "Adding 'about' after 'discuss' mirrors Hindi 'ke baare mein' "
                                 "— English 'discuss' already includes the meaning of 'about'.",
                "having_own": "Using 'is having' for possession (Hindi continuous tense transfer) "
                              "— English uses simple 'has' for stative possession.",
                "knowing": "Using 'am knowing' for a stative verb — Hindi allows continuous "
                           "with states ('jaan raha hoon'), English does not.",
                "works_uncountable": "Treating an uncountable noun as countable ('works') — "
                                     "a common Indian English pattern.",
                "present_perfect_past": "Present perfect with a finished-time word like 'yesterday' — "
                                        "Hindi allows this, English requires simple past.",
                "missing_article": "Articles (a/an/the) are missing — many Indian languages "
                                   "don't have articles, so this is a natural transfer.",
                "redundant_preposition": "Extra preposition after a verb — "
                                         "many of these verbs in Hindi take a postposition.",
                "subject_verb_number": "Subject-verb number mismatch — "
                                       "Hindi verbs agree with subject in number/gender differently.",
            }
            return explanations.get(name, f"L1 transfer pattern detected: {name}")
    return ""


def _detect_terminology_confusion(sa: str, ca: str, query: str,
                                  topic: str) -> tuple | None:
    """Return (explanation, root_cause, remediation) or None."""
    for term_set, remediation in _TERM_CONFUSIONS:
        sa_words = set(_tokenize(sa))
        ca_words = set(_tokenize(ca))
        in_sa = term_set & sa_words
        in_ca = term_set & ca_words
        if in_sa and in_ca and in_sa != in_ca:
            confused = in_sa - in_ca
            correct = in_ca - in_sa
            return (
                f"Using '{', '.join(confused)}' where '{', '.join(correct)}' is correct — "
                f"these terms refer to different concepts.",
                "linguistic_terminology",
                remediation,
            )
    return None


def _detect_overgeneralisation(sa: str, ca: str) -> str:
    """Detect if the student over-applied a regular pattern to an irregular case."""
    # Irregular past tense overgeneralisation: "goed" instead of "went"
    over_patterns = [
        ("goed", "went", "Using 'goed' instead of 'went' — you applied the regular -ed rule to an irregular verb."),
        ("runned", "ran", "Using 'runned' instead of 'ran'."),
        ("swimmed", "swam", "Using 'swimmed' instead of 'swam'."),
        ("breaked", "broke", "Using 'breaked' instead of 'broke'."),
        ("catched", "caught", "Using 'catched' instead of 'caught'."),
        ("mans", "men", "Using 'mans' instead of 'men' — irregular plural."),
        ("childs", "children", "Using 'childs' instead of 'children'."),
        ("mouses", "mice", "Using 'mouses' instead of 'mice'."),
        ("more bigger", "bigger", "Double comparative — either 'more big' or 'bigger', not both."),
        ("most biggest", "biggest", "Double superlative — either 'most big' or 'biggest', not both."),
    ]
    for error_form, correct_form, explanation in over_patterns:
        if error_form in sa:
            return explanation
    # General check: -ed suffix where correct answer uses a shorter form
    # (could indicate regularisation of irregular form)
    if sa.endswith("ed") and not ca.endswith("ed") and len(sa) >= len(ca) + 2:
        return (f"Possible overgeneralisation: applying the regular -ed rule. "
                f"The correct form '{ca}' is irregular — it doesn't follow the -ed pattern.")
    return ""


def _detect_shortcut(sa: str, query: str) -> bool:
    """Detect answers that use surface-level keyword matching."""
    if not query:
        return False
    # If the answer is a single word that matches a keyword in the query
    # but the answer should be a concept/explanation, it's likely a shortcut
    q_words = set(_tokenize(query.lower()))
    a_words = set(_tokenize(sa))
    if len(a_words) == 1 and a_words & q_words:
        return True
    # Very short answer to a long query suggests shortcut
    if len(sa.split()) < 3 and len(query.split()) > 10:
        return True
    return False


def _tokenize(text: str) -> list[str]:
    import re
    return [w.lower() for w in re.findall(r"[a-zA-Z]+", text)]


# ── Quick coarse check (already deterministic, unchanged logic) ────

def quick_coarse_check(student_answer: str, correct_answer: str,
                       response_ms: float = 0, confidence_1to5: int = 0) -> ErrorDiagnosis:
    """Fast deterministic heuristic — catches obvious patterns."""
    sa = student_answer.strip().lower()
    ca = correct_answer.strip().lower()

    if not sa:
        return ErrorDiagnosis(
            primary_error="conceptual_gap",
            explanation="Student left answer blank.",
            root_cause_topic="unknown",
            remediation="Build confidence with simpler warm-up problems before full difficulty.",
            confidence_calibration="underconfident",
            is_valid=True,
        )

    if sa == ca:
        return ErrorDiagnosis(primary_error="no_error", is_valid=True)

    # Impulsive: very fast wrong answer
    if response_ms > 0 and response_ms < 10000 and sa != ca:
        return ErrorDiagnosis(
            primary_error="impulsive",
            explanation=f"Answered in {response_ms/1000:.1f}s — likely impulsive without verification.",
            root_cause_topic="metacognition",
            remediation="Slow down. Count to 3 before submitting. Verify at least one check.",
            error_signature="impulsive::fast_wrong",
            is_valid=True,
        )

    return ErrorDiagnosis()  # Needs deeper classification
