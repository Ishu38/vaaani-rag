"""Active-learning exercises on the graph.

Active learning = the child CONSTRUCTS and SELF-CORRECTS with immediate feedback,
instead of receiving. Two modes:

  • Fix it   — the child DETECTS an error (analyse/evaluate, higher-order) in a
               real Indian-English transfer sentence, then sees why.
  • Build it — the child PRODUCES a sentence using words from their own graph.

Fix-it is fully deterministic (authored transfer bank) so it is instant and
reliable for a young learner; Build-it checks the target words are used and the
attempt is a real sentence, and rewards the act of constructing.
"""
import random

# Indian-English L1-transfer errors (pan-Hindi/Bengali). Each has ONE clearly
# tappable culprit word so "spot the slip" has an unambiguous answer.
FIXIT = [
    {"id": "know",  "words": ["I", "am", "knowing", "the", "answer", "."], "error": 2,
     "correct": "I know the answer.",
     "why": "‘Know’ names a state, not an action — state verbs (know, want, like) don’t take ‘-ing’."},
    {"id": "have",  "words": ["She", "is", "having", "a", "car", "."], "error": 2,
     "correct": "She has a car.",
     "why": "‘Have’ for owning is a state — not ‘is having’. ‘Having’ is only for actions like ‘having lunch’."},
    {"id": "does",  "words": ["He", "do", "his", "homework", "daily", "."], "error": 1,
     "correct": "He does his homework daily.",
     "why": "With he / she / it, the verb takes an ‘-s’: he does, she goes, it runs."},
    {"id": "work",  "words": ["I", "have", "too", "much", "works", "today", "."], "error": 4,
     "correct": "I have too much work today.",
     "why": "‘Work’ is uncountable — no ‘-s’ and no ‘a work’. Say ‘much work’, ‘some work’."},
    {"id": "good",  "words": ["What", "is", "your", "good", "name", "?"], "error": 3,
     "correct": "What is your name?",
     "why": "‘Good name’ is a translation of ‘shubh naam’. In English it’s just ‘your name’."},
    {"id": "yest",  "words": ["I", "have", "seen", "him", "yesterday", "."], "error": 1,
     "correct": "I saw him yesterday.",
     "why": "With a finished time like ‘yesterday’, use the simple past (saw), not ‘have seen’."},
    {"id": "disc",  "words": ["Let", "us", "discuss", "about", "the", "plan", "."], "error": 3,
     "correct": "Let us discuss the plan.",
     "why": "‘Discuss’ already means ‘talk about’ — you don’t add ‘about’ after it."},
    {"id": "myself", "words": ["Myself", "Rahul", "."], "error": 0,
     "correct": "I am Rahul.",
     "why": "To introduce yourself, say ‘I am Rahul’ or ‘My name is Rahul’ — not ‘Myself Rahul’."},
]
_BY_ID = {c["id"]: c for c in FIXIT}


def fixit_next(exclude: str | None = None) -> dict:
    """A challenge with the error HIDDEN — the child must find it."""
    pool = [c for c in FIXIT if c["id"] != exclude] or FIXIT
    c = random.choice(pool)
    return {"id": c["id"], "words": c["words"], "prompt": "Tap the word that isn’t right."}


def fixit_check(cid: str, idx: int) -> dict:
    c = _BY_ID.get(cid)
    if not c:
        return {"error": "unknown challenge"}
    ok = (idx == c["error"])
    return {
        "correct": ok,
        "error_index": c["error"],
        "error_word": c["words"][c["error"]],
        "correct_sentence": c["correct"],
        "why": c["why"],
        "feedback": (f"Yes! ‘{c['words'][c['error']]}’ is the slip." if ok
                     else "Not that one — look again. Which word breaks English grammar?"),
    }


def build_check(sentence: str, targets: list[str]) -> dict:
    """Reward constructing a real sentence that uses the child's chosen words."""
    s = (sentence or "").strip()
    low = s.lower()
    used = [t for t in targets if t.lower() in low]
    missing = [t for t in targets if t.lower() not in low]
    words = [w for w in s.replace(".", " ").split() if w]
    ok = (not missing) and len(words) >= 3
    msg = []
    if missing:
        msg.append(f"Use {', '.join(missing)} in your sentence too.")
    if len(words) < 3:
        msg.append("Make it a full sentence — at least a few words.")
    if ok:
        msg.append("Nice — you used your words in a real sentence! ⭐ Your stars just grew.")
    return {"ok": ok, "used": used, "missing": missing,
            "feedback": " ".join(msg) or "Keep going!"}
