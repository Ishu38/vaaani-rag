"""Piper-TTS narration pipeline.

Pulls full text for an ingested document, splits it into paragraph-sized
utterances, synthesizes each via piper, concatenates the WAVs, encodes to
MP3, and caches the result by SHA1 of (doc_signature + voice + version).
A cache hit returns in microseconds; a cold synth on a 2 vCPU box runs at
roughly 30-50x realtime, so a 5-minute article is ready in ~8s.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from config import (
    AUDIO_DIR,
    METADATA_PATH,
    PIPER_BIN,
    TTS_DEFAULT_VOICE,
    TTS_MP3_BITRATE,
    TTS_PARAGRAPH_CHARS,
    TTS_VOICES_DIR,
)

# Cache-key version. Bump when synth params or chunking change so old
# MP3s are regenerated transparently on next request.
CACHE_VERSION = "v1"

# Podcast crossfade duration between consecutive turns. Just enough to mask
# the silence at WAV boundaries without slurring words.
CROSSFADE_MS = 180

# Default speaker→voice mapping for podcast mode. Override via env
# VAAANI_TTS_PODCAST_VOICES="aria=<name>,rohan=<name>".
DEFAULT_PODCAST_VOICES = {
    "aria": "en_GB-jenny_dioco-medium",
    "rohan": "en_GB-alan-medium",
}


@dataclass
class NarrationResult:
    cache_hash: str
    mp3_path: Path
    duration_s: float
    voice: str
    doc_name: str
    cached: bool


@dataclass
class _Voice:
    name: str
    model: Path
    config: Path


def available_voices() -> list[str]:
    """Voice files dropped into data/tts_voices/. A voice is valid when both
    `<name>.onnx` and `<name>.onnx.json` exist."""
    out: list[str] = []
    for onnx in sorted(TTS_VOICES_DIR.glob("*.onnx")):
        if onnx.with_suffix(".onnx.json").exists():
            out.append(onnx.stem)
    return out


def _resolve_voice(name: str | None) -> _Voice:
    requested = name or TTS_DEFAULT_VOICE
    model = TTS_VOICES_DIR / f"{requested}.onnx"
    config = TTS_VOICES_DIR / f"{requested}.onnx.json"
    if not model.exists() or not config.exists():
        installed = ", ".join(available_voices()) or "(none)"
        raise FileNotFoundError(
            f"voice '{requested}' not found in {TTS_VOICES_DIR}. installed: {installed}"
        )
    return _Voice(name=requested, model=model, config=config)


def _load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {"files": {}, "chunks": []}
    return json.loads(METADATA_PATH.read_text())


def list_narratable_docs() -> list[dict]:
    """Docs we can narrate: ingested files with at least one chunk of text."""
    meta = _load_metadata()
    by_source: dict[str, dict] = {}
    for ch in meta.get("chunks", []):
        src = ch.get("source")
        if not src:
            continue
        slot = by_source.setdefault(src, {"chars": 0, "chunks": 0})
        slot["chars"] += len(ch.get("text", ""))
        slot["chunks"] += 1
    out: list[dict] = []
    for path, info in meta.get("files", {}).items():
        name = info.get("name") or Path(path).name
        stat = by_source.get(name, {"chars": 0, "chunks": 0})
        if stat["chars"] == 0:
            continue
        # Rough est: 150 wpm narration, ~5 chars/word.
        est_minutes = stat["chars"] / (150 * 5)
        out.append(
            {
                "doc_name": name,
                "signature": info.get("signature", ""),
                "chunks": stat["chunks"],
                "chars": stat["chars"],
                "estimated_minutes": round(est_minutes, 1),
            }
        )
    out.sort(key=lambda d: d["doc_name"].lower())
    return out


def _gather_doc_text(doc_name: str) -> tuple[str, str]:
    """Returns (concatenated_text, doc_signature). Raises KeyError if doc unknown."""
    meta = _load_metadata()
    pieces: list[tuple[int, str]] = []
    for ch in meta.get("chunks", []):
        if ch.get("source") == doc_name:
            pieces.append((ch.get("chunk_no", 0), ch.get("text", "")))
    if not pieces:
        raise KeyError(f"no ingested text for doc '{doc_name}'")
    pieces.sort(key=lambda p: p[0])
    text = "\n\n".join(p[1].strip() for p in pieces if p[1].strip())

    signature = ""
    for _path, info in meta.get("files", {}).items():
        if info.get("name") == doc_name:
            signature = info.get("signature", "")
            break
    return text, signature


def _split_paragraphs(text: str, cap: int = TTS_PARAGRAPH_CHARS) -> list[str]:
    # Normalize newlines, drop control whitespace except \n.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    out: list[str] = []
    for para in raw_paras:
        if len(para) <= cap:
            out.append(para)
            continue
        # Split overlong paragraphs on sentence boundaries.
        sents = re.split(r"(?<=[.!?])\s+", para)
        buf = ""
        for s in sents:
            if buf and len(buf) + 1 + len(s) > cap:
                out.append(buf)
                buf = s
            else:
                buf = f"{buf} {s}".strip() if buf else s
        if buf:
            out.append(buf)
    return out


def _cache_hash(doc_signature: str, voice: str, body_hash: str) -> str:
    raw = f"{CACHE_VERSION}|{voice}|{doc_signature}|{body_hash}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def cache_path_for(cache_hash: str) -> Path:
    return AUDIO_DIR / f"{cache_hash}.mp3"


def _piper_synth(text: str, voice: _Voice, wav_out: Path) -> None:
    """Run piper as a subprocess for a single utterance."""
    piper_path = shutil.which(PIPER_BIN) or PIPER_BIN
    cmd = [piper_path, "-m", str(voice.model), "-c", str(voice.config), "-f", str(wav_out)]
    proc = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        capture_output=True,
        timeout=180,
    )
    if proc.returncode != 0 or not wav_out.exists() or wav_out.stat().st_size < 200:
        err = proc.stderr.decode("utf-8", errors="replace")[-400:]
        raise RuntimeError(f"piper synth failed ({proc.returncode}): {err}")


def _concat_to_mp3(wav_paths: list[Path], mp3_out: Path) -> None:
    """ffmpeg concat-demuxer → libmp3lame, mono, narration bitrate."""
    if not wav_paths:
        raise ValueError("no WAVs to concat")
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as listing:
        for wav in wav_paths:
            listing.write(f"file '{wav.as_posix()}'\n")
        listing_path = Path(listing.name)
    try:
        cmd = [
            ffmpeg,
            "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(listing_path),
            "-ac", "1",
            "-codec:a", "libmp3lame",
            "-b:a", TTS_MP3_BITRATE,
            "-y", str(mp3_out),
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
        if proc.returncode != 0 or not mp3_out.exists():
            err = proc.stderr.decode("utf-8", errors="replace")[-400:]
            raise RuntimeError(f"ffmpeg concat failed ({proc.returncode}): {err}")
    finally:
        listing_path.unlink(missing_ok=True)


def _probe_duration_s(mp3_path: Path) -> float:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(mp3_path)],
        capture_output=True, timeout=15,
    )
    try:
        return float(proc.stdout.decode("utf-8").strip())
    except ValueError:
        return 0.0


def narrate_text(
    text: str,
    *,
    voice: str | None = None,
    doc_signature: str = "",
    doc_name: str = "raw",
) -> NarrationResult:
    """Synthesize arbitrary text. Caches by (doc_signature, voice, text hash)."""
    if not text.strip():
        raise ValueError("empty text")
    v = _resolve_voice(voice)
    body_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    sig = doc_signature or body_hash
    chash = _cache_hash(sig, v.name, body_hash)
    mp3_path = cache_path_for(chash)
    if mp3_path.exists() and mp3_path.stat().st_size > 1000:
        return NarrationResult(
            cache_hash=chash,
            mp3_path=mp3_path,
            duration_s=_probe_duration_s(mp3_path),
            voice=v.name,
            doc_name=doc_name,
            cached=True,
        )

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        raise ValueError("no narratable paragraphs after splitting")

    with tempfile.TemporaryDirectory(prefix="vaaani-tts-") as tmpdir:
        tmp = Path(tmpdir)
        wavs: list[Path] = []
        for i, para in enumerate(paragraphs):
            wav_out = tmp / f"part_{i:04d}.wav"
            _piper_synth(para, v, wav_out)
            wavs.append(wav_out)
        _concat_to_mp3(wavs, mp3_path)

    return NarrationResult(
        cache_hash=chash,
        mp3_path=mp3_path,
        duration_s=_probe_duration_s(mp3_path),
        voice=v.name,
        doc_name=doc_name,
        cached=False,
    )


def narrate_doc(doc_name: str, *, voice: str | None = None) -> NarrationResult:
    text, signature = _gather_doc_text(doc_name)
    return narrate_text(
        text,
        voice=voice,
        doc_signature=signature,
        doc_name=doc_name,
    )


# =========================================================================
#  PODCAST MODE — 2-voice dialogue with crossfaded concat
# =========================================================================
import os  # noqa: E402  (kept near the podcast block for locality)

from .script_writer import Turn, generate_script  # noqa: E402


def _resolve_podcast_voices() -> dict[str, _Voice]:
    """Speaker→Voice map. Reads VAAANI_TTS_PODCAST_VOICES if set, else default."""
    mapping = dict(DEFAULT_PODCAST_VOICES)
    override = os.environ.get("VAAANI_TTS_PODCAST_VOICES", "").strip()
    if override:
        for pair in override.split(","):
            if "=" not in pair:
                continue
            k, v = pair.split("=", 1)
            k = k.strip().lower()
            v = v.strip()
            if k in mapping and v:
                mapping[k] = v
    return {speaker: _resolve_voice(name) for speaker, name in mapping.items()}


def _concat_with_crossfade(wav_paths: list[Path], mp3_out: Path) -> None:
    """ffmpeg `acrossfade` pairwise across all turns → mono MP3.

    The filter chain crossfades pairs left-to-right. For N inputs we build
    N-1 acrossfade nodes. A short fade smooths the click between
    Piper utterances without sounding artificial.
    """
    if not wav_paths:
        raise ValueError("no WAVs to concat")
    if len(wav_paths) == 1:
        # Single turn — just transcode to MP3, no fades needed.
        _concat_to_mp3(wav_paths, mp3_out)
        return

    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    fade_s = CROSSFADE_MS / 1000.0
    inputs: list[str] = []
    for wav in wav_paths:
        inputs.extend(["-i", str(wav)])

    # Build filter graph: [0][1]acrossfade=d=0.18:c1=tri:c2=tri[a1];
    # [a1][2]acrossfade=...[a2]; ...
    nodes: list[str] = []
    prev = "[0]"
    for i in range(1, len(wav_paths)):
        out_label = f"[a{i}]" if i < len(wav_paths) - 1 else "[out]"
        nodes.append(
            f"{prev}[{i}]acrossfade=d={fade_s}:c1=tri:c2=tri{out_label}"
        )
        prev = out_label
    filtergraph = ";".join(nodes)

    cmd = [
        ffmpeg,
        "-hide_banner", "-loglevel", "error",
        *inputs,
        "-filter_complex", filtergraph,
        "-map", "[out]",
        "-ac", "1",
        "-codec:a", "libmp3lame",
        "-b:a", TTS_MP3_BITRATE,
        "-y", str(mp3_out),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=600)
    if proc.returncode != 0 or not mp3_out.exists():
        err = proc.stderr.decode("utf-8", errors="replace")[-400:]
        raise RuntimeError(f"ffmpeg crossfade failed ({proc.returncode}): {err}")


def _podcast_cache_hash(doc_signature: str, voices: dict[str, _Voice], script_hash: str) -> str:
    voice_part = ",".join(f"{k}:{v.name}" for k, v in sorted(voices.items()))
    raw = f"{CACHE_VERSION}|podcast|{voice_part}|{doc_signature}|{script_hash}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def podcast_doc(doc_name: str) -> NarrationResult:
    """Turn an ingested document into a 2-host podcast MP3 (cached)."""
    text, signature = _gather_doc_text(doc_name)
    voices = _resolve_podcast_voices()

    # Cache-key uses the SOURCE signature (not the generated script), so the
    # same doc + same voices always reuses the previous synth. We compute
    # the script lazily only on cache miss.
    speaker_part = ",".join(f"{k}:{v.name}" for k, v in sorted(voices.items()))
    pre_hash = hashlib.sha1(
        f"{CACHE_VERSION}|podcast-pre|{speaker_part}|{signature}".encode("utf-8")
    ).hexdigest()[:16]
    mp3_path = cache_path_for(pre_hash)
    if mp3_path.exists() and mp3_path.stat().st_size > 1000:
        return NarrationResult(
            cache_hash=pre_hash,
            mp3_path=mp3_path,
            duration_s=_probe_duration_s(mp3_path),
            voice="podcast",
            doc_name=doc_name,
            cached=True,
        )

    turns: list[Turn] = generate_script(text, title=doc_name)

    with tempfile.TemporaryDirectory(prefix="vaaani-podcast-") as tmpdir:
        tmp = Path(tmpdir)
        wavs: list[Path] = []
        for i, turn in enumerate(turns):
            voice = voices.get(turn.speaker)
            if voice is None:
                continue
            wav_out = tmp / f"turn_{i:04d}_{turn.speaker}.wav"
            _piper_synth(turn.text, voice, wav_out)
            wavs.append(wav_out)
        if not wavs:
            raise RuntimeError("podcast script produced no synthesizable turns")
        _concat_with_crossfade(wavs, mp3_path)

    return NarrationResult(
        cache_hash=pre_hash,
        mp3_path=mp3_path,
        duration_s=_probe_duration_s(mp3_path),
        voice="podcast",
        doc_name=doc_name,
        cached=False,
    )
