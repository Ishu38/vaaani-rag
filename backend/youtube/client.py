"""Tiny wrapper over youtube-transcript-api.

Job: take a YouTube URL or video id from a student, return a markdown
document ready to drop into data/raw/ for the existing ingest pipeline.
No API key needed; the library scrapes the public timed-text endpoint.

We try a sensible language preference order for Indian students:
English first, then Hindi, then Bengali, then anything. The library
falls back across the list when a transcript isn't available in the
first choice — Indian YouTube tutors commonly post in Hindi or mix
languages, so we'd rather have a hin transcript than no transcript.
"""
from __future__ import annotations

import re
from datetime import timedelta

# Defer the actual API client construction so module import is cheap.

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_URL_PATTERNS = [
    re.compile(r"(?:youtube\.com/watch\?(?:.*&)?v=)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtu\.be/)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/live/)([A-Za-z0-9_-]{11})"),
]

DEFAULT_LANGS = ("en", "en-IN", "en-US", "en-GB", "hi", "bn", "ta", "te", "mr", "kn", "ml", "gu", "pa")


class TranscriptError(Exception):
    """No transcript available, video unavailable, or YouTube refused us."""


def parse_video_id(url_or_id: str) -> str:
    """Accept any of: plain 11-char video id, youtu.be/<id>, watch?v=<id>,
    embed/<id>, shorts/<id>, live/<id>. Raises ValueError if nothing
    matches."""
    s = (url_or_id or "").strip()
    if not s:
        raise ValueError("empty input")
    # Bare id
    if _VIDEO_ID_RE.match(s):
        return s
    for pat in _URL_PATTERNS:
        m = pat.search(s)
        if m:
            return m.group(1)
    raise ValueError("Could not extract a YouTube video id from that input")


def _fmt_timestamp(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    total = int(td.total_seconds())
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fetch_transcript_markdown(video_id: str, *, languages: tuple[str, ...] = DEFAULT_LANGS) -> tuple[str, str]:
    """Return (title-ish header, markdown body) for one YouTube video.

    The header is just `# YouTube · <video_id>` — we deliberately don't
    fetch oEmbed for the real video title because that adds a network
    hop and a third-party dependency on YouTube's metadata endpoint.
    The student can rename the resulting file if they want.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import (
            TranscriptsDisabled, NoTranscriptFound, VideoUnavailable,
            CouldNotRetrieveTranscript,
        )
    except ImportError as e:
        raise TranscriptError(f"youtube-transcript-api not installed: {e}")

    api = YouTubeTranscriptApi()
    try:
        transcript = api.fetch(video_id, languages=list(languages))
    except VideoUnavailable:
        raise TranscriptError("video is unavailable or private")
    except TranscriptsDisabled:
        raise TranscriptError("transcripts are disabled on this video")
    except NoTranscriptFound:
        raise TranscriptError("no transcript in any of the supported languages")
    except CouldNotRetrieveTranscript as e:
        raise TranscriptError(f"YouTube refused: {e}")
    except Exception as e:
        raise TranscriptError(f"transcript fetch failed: {e}")

    snippets = list(transcript.snippets)
    if not snippets:
        raise TranscriptError("transcript came back empty")

    # Build paragraphs by clustering snippets into ~2-minute chunks with
    # a leading timestamp so the student can scrub back to the source
    # video if they want to verify. Each cluster becomes a paragraph,
    # which is also what our chunker splits on.
    cluster_seconds = 120
    paragraphs: list[str] = []
    current_start = snippets[0].start
    current_text: list[str] = []
    for snip in snippets:
        if snip.start - current_start > cluster_seconds and current_text:
            paragraphs.append(f"[{_fmt_timestamp(current_start)}] " + " ".join(current_text).strip())
            current_text = []
            current_start = snip.start
        text = (snip.text or "").replace("\n", " ").strip()
        if text:
            current_text.append(text)
    if current_text:
        paragraphs.append(f"[{_fmt_timestamp(current_start)}] " + " ".join(current_text).strip())

    body = "\n\n".join(paragraphs)
    header = f"YouTube · {video_id}"
    return header, body
