"""Convert an article into a 2-host podcast dialogue via DeepSeek.

The hosts are fixed (Aria = curious learner, Rohan = subject-matter explainer).
Both speakers are constrained to the article's content; no fabrications, no
new examples drawn from outside the text. The prompt asks DeepSeek to return
strict JSON so we can drive per-turn TTS deterministically.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from llm import call_deepseek


@dataclass(frozen=True)
class Turn:
    speaker: str   # "aria" or "rohan"
    text: str


SYSTEM_PROMPT = (
    "You convert study material into a two-host podcast script. The hosts are:\n"
    "- Aria: a curious, sharp learner who asks clarifying questions and "
    "restates ideas in plain language. Warm, conversational, occasionally "
    "humorous.\n"
    "- Rohan: a patient subject expert who explains the material, gives "
    "concrete examples drawn ONLY from the article, and corrects "
    "misunderstandings. Calm, precise, never lectures for more than 3-4 "
    "sentences without inviting Aria back in.\n\n"
    "Hard rules:\n"
    "1. Use ONLY facts, examples, and quotes from the article. Do not invent "
    "names, statistics, anecdotes, or sources. If the article is sparse, the "
    "podcast is short.\n"
    "2. Alternate speakers naturally. Open with Aria framing what they're "
    "about to discuss. Close with Aria offering a one-sentence takeaway.\n"
    "3. Keep each turn under 80 words. Aim for 8-16 turns total depending on "
    "article length.\n"
    "4. Plain spoken English. Spell out symbols (say \"theta\" not \"θ\", "
    "\"squared\" not \"^2\"). Numbers as words for low integers, numerals "
    "for the rest. No markdown, no asterisks, no headings.\n"
    "5. Output strict JSON with this shape:\n"
    '   {"turns": [{"speaker": "aria"|"rohan", "text": "..."}, ...]}\n'
    "Nothing else. No prose before or after the JSON."
)

USER_TEMPLATE = (
    "Article title: {title}\n\n"
    "Article body:\n{body}\n\n"
    "Generate the podcast script as JSON now."
)


def _truncate_body(body: str, char_cap: int) -> str:
    """Keep prompts within DeepSeek's context budget. For very long sources
    we trim trailing material — the open paragraphs are usually the densest
    introduction, and the podcast format favours breadth over completeness."""
    if len(body) <= char_cap:
        return body
    return body[:char_cap].rsplit(" ", 1)[0] + " […]"


def generate_script(
    body: str,
    *,
    title: str = "Untitled",
    max_body_chars: int = 18000,
) -> list[Turn]:
    """Call DeepSeek and parse the response into ordered Turn objects."""
    if not body.strip():
        raise ValueError("empty article body")

    trimmed = _truncate_body(body, max_body_chars)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(title=title, body=trimmed)},
    ]
    resp = call_deepseek(messages, stream=False, json_mode=True)
    if not isinstance(resp, dict):
        raise RuntimeError("DeepSeek returned no response")
    try:
        content = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"unexpected DeepSeek shape: {e}")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"DeepSeek did not return JSON: {e}")

    raw_turns = parsed.get("turns") or []
    if not raw_turns:
        raise RuntimeError("DeepSeek returned an empty turn list")

    out: list[Turn] = []
    for t in raw_turns:
        spk = (t.get("speaker") or "").strip().lower()
        txt = (t.get("text") or "").strip()
        if spk not in {"aria", "rohan"} or not txt:
            continue
        out.append(Turn(speaker=spk, text=txt))
    if not out:
        raise RuntimeError("no usable turns after filtering")
    return out
