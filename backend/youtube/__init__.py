"""YouTube transcript ingest. Paste a YouTube URL, get a doc in the corpus."""
from .client import (
    TranscriptError,
    fetch_transcript_markdown,
    parse_video_id,
)

__all__ = [
    "TranscriptError",
    "fetch_transcript_markdown",
    "parse_video_id",
]
