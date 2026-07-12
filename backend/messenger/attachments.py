"""Attachment handling for messenger bots: photos → OCR → ingest, PDFs → ingest.

Transport-agnostic. Each transport (telegram.py, future whatsapp.py)
downloads the binary to a local path and calls into here with the path
plus a friendly source label. We OCR / parse, persist a readable doc
into data/raw/, and run the existing ingest() pipeline so the new
material shows up in chat, Review, Feynman, and Anki immediately.

CPU-only on purpose: Tesseract subprocess + PyPDF text extract. No
cloud OCR APIs, no GPU model. Matches the moat described in
[[feedback_vaani_cpu_centric_moat]].
"""
from __future__ import annotations

import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from config import INDEX_PATH, METADATA_PATH, RAW_DIR
from ingest import ingest

# Languages to try when OCRing. Indian classrooms are routinely mixed
# script — adding `hin+ben` costs ~20% CPU but catches Devanagari and
# Bangla pages students will absolutely send. Override per call if a
# transport ever wants to specialise.
DEFAULT_LANGS = "eng+hin+ben"

# Telegram strips EXIF and reduces the largest photo to ~1280x1280, which
# is fine for Tesseract. We still cap the OCR time to keep an angry
# blurry photo from pegging the VPS for minutes.
OCR_TIMEOUT_S = 120

# Minimum recognised characters for us to treat the photo as a real
# document worth ingesting. Random snaps of a desk return ~5 characters
# of noise from Tesseract; we want at least a sentence.
OCR_MIN_CHARS = 40


# =========================================================================
#  Tesseract availability
# =========================================================================

def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def tesseract_languages() -> list[str]:
    """Languages installed locally (eng, hin, ben, etc.)."""
    binary = shutil.which("tesseract")
    if not binary:
        return []
    try:
        proc = subprocess.run([binary, "--list-langs"], capture_output=True, timeout=10)
    except Exception:
        return []
    out = (proc.stdout or b"").decode("utf-8", errors="replace") + (proc.stderr or b"").decode("utf-8", errors="replace")
    langs: list[str] = []
    for line in out.splitlines():
        line = line.strip()
        if line and " " not in line and not line.endswith(":"):
            langs.append(line)
    return langs


# =========================================================================
#  Photo OCR
# =========================================================================

def ocr_image(image_path: Path, languages: str = DEFAULT_LANGS) -> str:
    """Run tesseract on a local image, return whatever text it extracted.
    Returns empty string on any failure — callers decide how to message
    the user, since the bot's "could not read" copy needs context."""
    if not tesseract_available():
        raise RuntimeError("tesseract is not installed on this server")
    # Fall back to eng-only if some requested language isn't installed —
    # otherwise tesseract aborts with "Error opening data file" and we
    # return zero text on a perfectly readable English page.
    installed = set(tesseract_languages())
    wanted = [lang for lang in languages.split("+") if lang in installed]
    if not wanted:
        wanted = ["eng"] if "eng" in installed else list(installed)[:1]
    lang_arg = "+".join(wanted) or "eng"

    try:
        proc = subprocess.run(
            ["tesseract", str(image_path), "-", "-l", lang_arg, "--psm", "6"],
            capture_output=True,
            timeout=OCR_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("OCR timed out — photo was too large or unclear")
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[-300:]
        raise RuntimeError(f"OCR failed ({proc.returncode}): {err}")
    text = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
    return _cleanup_ocr_text(text)


_WS_RUN = re.compile(r"[ \t]+")
_HARD_WRAPS = re.compile(r"\n{3,}")


def _cleanup_ocr_text(text: str) -> str:
    """Tesseract returns a mid-paragraph linebreak after each scanned
    line. Collapse those into spaces so the chunker (which splits on
    paragraphs) sees proper paragraphs."""
    lines = text.splitlines()
    paragraphs: list[str] = []
    buf: list[str] = []
    for ln in lines:
        ln = _WS_RUN.sub(" ", ln).strip()
        if not ln:
            if buf:
                paragraphs.append(" ".join(buf))
                buf = []
            continue
        buf.append(ln)
    if buf:
        paragraphs.append(" ".join(buf))
    out = "\n\n".join(paragraphs)
    return _HARD_WRAPS.sub("\n\n", out).strip()


# =========================================================================
#  PDF passthrough
# =========================================================================

def is_pdf(mime_type: str | None, filename: str | None) -> bool:
    if mime_type and mime_type.lower() == "application/pdf":
        return True
    if filename and filename.lower().endswith(".pdf"):
        return True
    return False


# =========================================================================
#  Persist + ingest
# =========================================================================

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_label(label: str) -> str:
    return _SAFE_NAME_RE.sub("-", (label or "").strip())[:40].strip("-") or "doc"


def _dest_dir(owner_user_id: int | None) -> Path:
    """Per-user subdir keeps file keys unique and ownership unambiguous."""
    d = RAW_DIR / f"u{owner_user_id}" if owner_user_id else RAW_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _record_owner(path: Path, owner_user_id: int | None) -> None:
    """Stamp privacy-scope ownership (see scope.py) for an ingested file."""
    if not owner_user_id:
        return
    try:
        import scope
        from auth import service as auth_service
        user = auth_service.get_user_by_id(owner_user_id)
        scope.record_ownership(
            str(path.resolve()), owner_user_id, scope.sharing_school_ids(user)
        )
    except Exception:
        pass


def save_text_as_doc(
    text: str, *, source_label: str, prefix: str = "telegram-photo",
    owner_user_id: int | None = None,
) -> Path:
    """Write OCRed (or otherwise text-extracted) content into data/raw/
    so the existing ingest pipeline picks it up. Returns the path."""
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    label = _safe_label(source_label)
    fname = f"{prefix}-{label}-{ts}.md"
    path = _dest_dir(owner_user_id) / fname
    front = f"# Photo from {source_label} · {ts}\n\n" if "photo" in prefix else f"# {source_label}\n\n"
    path.write_text(front + text, encoding="utf-8")
    _record_owner(path, owner_user_id)
    return path


def save_pdf(
    source_path: Path, *, source_label: str, prefix: str = "telegram-doc",
    owner_user_id: int | None = None,
) -> Path:
    """Copy an inbound PDF into data/raw/ with a stable, sanitised name."""
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    label = _safe_label(source_label)
    fname = f"{prefix}-{label}-{ts}.pdf"
    dest = _dest_dir(owner_user_id) / fname
    shutil.copy2(source_path, dest)
    _record_owner(dest, owner_user_id)
    return dest


def run_ingest_and_refresh() -> dict:
    """Re-run the existing pipeline; bust caches so new chunks/nodes
    surface in chat, Review, Feynman, and audio immediately."""
    summary = ingest(RAW_DIR, INDEX_PATH, METADATA_PATH)
    try:
        from main import retriever as _retriever  # lazy to dodge cycle
        _retriever.reload()
    except Exception:
        pass
    try:
        from adaptive import spaced as _spaced
        _spaced.invalidate_graph_cache()
        _spaced.invalidate_chunks_cache()
    except Exception:
        pass
    return summary


# =========================================================================
#  Top-level entry points used by the dispatcher
# =========================================================================

def ingest_photo(
    image_path: Path, *, source_label: str, caption: str = "",
    owner_user_id: int | None = None,
) -> dict:
    """OCR + persist + ingest. Returns a dict the dispatcher can format
    into a reply: {chars, words, chunks_added, total_chunks, doc_filename}."""
    text = ocr_image(image_path)
    if len(text) < OCR_MIN_CHARS:
        return {
            "ok": False,
            "reason": "too_little_text",
            "chars": len(text),
        }
    full = text if not caption else f"{caption.strip()}\n\n{text}"
    doc_path = save_text_as_doc(full, source_label=source_label, owner_user_id=owner_user_id)
    summary = run_ingest_and_refresh()
    return {
        "ok": True,
        "doc_filename": doc_path.name,
        "chars": len(full),
        "words": len(full.split()),
        "chunks_added": summary.get("chunks_added", 0),
        "total_chunks": summary.get("total_chunks", 0),
    }


def ingest_pdf(pdf_path: Path, *, source_label: str, owner_user_id: int | None = None) -> dict:
    """Drop an inbound PDF into data/raw/ + ingest. PyPDF (already a
    dep of the existing ingest pipeline) handles the text extraction."""
    saved = save_pdf(pdf_path, source_label=source_label, owner_user_id=owner_user_id)
    summary = run_ingest_and_refresh()
    return {
        "ok": True,
        "doc_filename": saved.name,
        "chunks_added": summary.get("chunks_added", 0),
        "total_chunks": summary.get("total_chunks", 0),
    }
