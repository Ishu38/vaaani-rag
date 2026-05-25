"""Audio narration of ingested documents via Piper TTS (CPU-only)."""
from .narrator import (
    NarrationResult,
    available_voices,
    cache_path_for,
    list_narratable_docs,
    narrate_doc,
    narrate_text,
    podcast_doc,
)

__all__ = [
    "NarrationResult",
    "available_voices",
    "cache_path_for",
    "list_narratable_docs",
    "narrate_doc",
    "narrate_text",
    "podcast_doc",
]
