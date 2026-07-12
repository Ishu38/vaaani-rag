"""Developmental output firewall — a streaming, boundary-safe scrubber that
suppresses phonetic notation (IPA) in generative output for young learners.

Why this exists (product + patent):
  • Product: Vaaani's standing rule is plain-language for every learner. A
    generative model can still emit IPA (e.g. "cat is /kæt/") mid-answer, which
    is meaningless — and discouraging — to a Grade-1 child. Authored templates
    avoid it; free-form LLM text does not. This closes that gap at the source.
  • Patent (Explore apparatus, claims 4–5): the "developmental output firewall"
    is claimed as a deterministic, streaming-boundary-safe mechanism that, as a
    function of a learner developmental-stage signal, suppresses a designated
    symbol class across the token boundaries of a model output stream. This is
    the reduction-to-practice of that mechanism.

Design:
  • Gate by GRADE. Below FIREWALL_GRADE_THRESHOLD (default 5 — the phonics gate,
    cf. units_sound.json `_g5_rules`) the firewall is active; at/above it, IPA
    passes through (older learners / linguistics use). Unknown grade fails safe
    to protected, because the audience is young L2 learners.
  • Two protected forms:
      1. Delimited transcriptions — /kæt/ or [kæt] — dropped only when the span
         actually contains an IPA-signature character (so "and/or", "km/h",
         "[1]", markdown "[text]" survive untouched).
      2. Bare IPA letters — the IPA Extensions block + stress/length marks —
         dropped individually. Deliberately conservative: Latin-1 letters that
         occur in ordinary words/names (æ, ç, ð, ø, œ) are NOT stripped as bare
         characters; they only count as evidence *inside* a delimited span.
  • Streaming safety: a hold-buffer withholds emission from an opening delimiter
    until the class can be resolved (closing delimiter arrives, the span grows
    past a bounded width, or the stream ends), so a transcription split across
    two token chunks — "…/k" then "æt/…" — is still caught. Latency is bounded
    by MAX_SPAN: text before any delimiter is emitted immediately.
"""
from __future__ import annotations

import os
import re

# Grade below which phonetic notation is suppressed. Bounded to a sane range so
# a mis-set env var can't disable protection for, say, Grade 1.
FIREWALL_GRADE_THRESHOLD = max(1, min(12, int(os.environ.get("VAAANI_FIREWALL_GRADE", "5"))))
# What to do when the learner's grade is unknown: "protect" (default) or "off".
FIREWALL_DEFAULT = os.environ.get("VAAANI_FIREWALL_DEFAULT", "protect").strip().lower()
# Max width of a held delimited span; beyond this it isn't a tight transcription.
MAX_SPAN = 48

# Dedicated IPA stress / length modifiers (outside the IPA Extensions block).
_IPA_MODS = {0x02C8, 0x02CC, 0x02D0, 0x02D1}  # ˈ ˌ ː ˑ
# Letters that live outside the IPA Extensions block but strongly signal a
# transcription when found *inside* a delimited span. NOT stripped bare.
_IPA_SPAN_EXTRA = set("θðŋæøœçβʍ")


def _is_ipa_bare(ch: str) -> bool:
    """True for characters we will strip even outside delimiters (unambiguous IPA)."""
    o = ord(ch)
    return (0x0250 <= o <= 0x02AF) or o in _IPA_MODS


def _is_ipa_span_char(ch: str) -> bool:
    """True for characters that mark a delimited span as phonetic."""
    return _is_ipa_bare(ch) or ch in _IPA_SPAN_EXTRA


def _span_is_phonetic(span: str) -> bool:
    return any(_is_ipa_span_char(c) for c in span) or _is_ascii_phoneme_span(span)


# A slash-delimited span of 1–6 ASCII letters (e.g. /f/, /m/, /ph/, /sh/) is a
# phoneme transcription in plain-ASCII form. The IPA-Unicode check above misses
# these because the letters are ordinary ASCII. For sub-notation-gate learners
# these must be scrubbed too — a Grade 2 child should never see /f/ in prose.
# Guard against false positives on paths/dates ("bin/bash", "2026/07/08",
# "and/or"): require a single letter-only token with no inner slash, space, or
# digit, length 1–6.
_ASCII_PHONEME_RE = re.compile(r"^/?[A-Za-z]{1,6}/?$")

def _is_ascii_phoneme_span(span: str) -> bool:
    inner = span.strip("/")
    if not inner or len(inner) > 6:
        return False
    if any(c in inner for c in " /\\0123456789"):
        return False
    return inner.isalpha()


def _strip_bare(s: str) -> str:
    return "".join("" if _is_ipa_bare(c) else c for c in s)


class Firewall:
    """Stateful streaming scrubber. Feed deltas; call flush() at end of stream."""

    def __init__(self, active: bool, substitute: str = "", max_span: int = MAX_SPAN):
        self.active = active
        self.sub = substitute
        self.max_span = max_span
        self._pending = ""  # unresolved tail held from an opening delimiter onward

    def feed(self, chunk: str) -> str:
        """Consume a chunk; return the safe text ready to emit now."""
        if not self.active:
            return chunk or ""
        if not chunk:
            return ""
        self._pending += chunk
        out, self._pending = self._consume(self._pending, final=False)
        return out

    def flush(self) -> str:
        """Resolve any held tail at end of stream (nothing is held after this)."""
        if not self.active:
            return ""
        out, rest = self._consume(self._pending, final=True)
        self._pending = ""
        return out + rest

    def _consume(self, text: str, final: bool) -> tuple[str, str]:
        out: list[str] = []
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if ch == "/" or ch == "[":
                close = "]" if ch == "[" else "/"
                j = text.find(close, i + 1)
                if j != -1 and (j - i) <= self.max_span:
                    span = text[i:j + 1]
                    out.append(self.sub if _span_is_phonetic(span) else span)
                    i = j + 1
                    continue
                # No usable closer yet.
                if final:
                    out.append(_strip_bare(text[i:]))
                    return ("".join(out), "")
                if (n - i) > self.max_span:
                    # Too wide to be a transcription — treat the delimiter as literal.
                    out.append(ch)
                    i += 1
                    continue
                # Hold from here; the closer may arrive in the next chunk.
                return ("".join(out), text[i:])
            out.append("" if _is_ipa_bare(ch) else ch)
            i += 1
        return ("".join(out), "")


# ── grade signal ────────────────────────────────────────────────────────────

def grade_from_age(age: int | None) -> int | None:
    """Rough grade from completed years (Grade 1 ≈ age 6). Clamped 1..12."""
    if age is None:
        return None
    return max(1, min(12, age - 5))


def resolve_grade(explicit_grade, user: dict | None) -> int | None:
    """Prefer an explicit request grade; else derive from the user's DOB."""
    if explicit_grade is not None:
        try:
            return int(explicit_grade)
        except (TypeError, ValueError):
            pass
    dob = (user or {}).get("date_of_birth")
    if dob:
        try:
            from auth.dpdp import age_from_dob
            g = grade_from_age(age_from_dob(dob))
            if g is not None:
                return g
        except Exception:
            pass
    return None


def is_active(grade: int | None) -> bool:
    if grade is None:
        return FIREWALL_DEFAULT != "off"  # fail-safe: protect when unknown
    return grade < FIREWALL_GRADE_THRESHOLD


def for_request(explicit_grade, user: dict | None) -> Firewall:
    """Build a firewall configured for this learner."""
    return Firewall(active=is_active(resolve_grade(explicit_grade, user)))


def scrub_text(text: str, *, explicit_grade=None, user: dict | None = None,
               active: bool | None = None) -> str:
    """One-shot scrub for non-streaming answers."""
    if active is None:
        active = is_active(resolve_grade(explicit_grade, user))
    fw = Firewall(active=active)
    return fw.feed(text) + fw.flush()
