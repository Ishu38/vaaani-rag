"""Document ingestion pipeline.

Reads .pdf/.txt/.md/.docx files, chunks them with overlap, embeds with
sentence-transformers, and stores the vectors in a TurboQuantIndex with a
JSON sidecar mapping each chunk position back to its source.

Incremental: files whose (path, mtime, size) are already recorded are skipped.

CLI:
    python ingest.py --source ./data/raw --index ./data/index.tq
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer
from turbovec import TurboQuantIndex

from community import build_communities, save_communities
from config import (
    BIT_WIDTH,
    CHUNK_OVERLAP,
    CHUNK_TOKENS,
    COMMUNITIES_PATH,
    EMBED_DIM,
    EMBED_MODEL_NAME,
    GRAPH_PATH,
    INDEX_PATH,
    METADATA_PATH,
    RAW_DIR,
)
from extractor import extract_chunk
from graph import KnowledgeGraph
import httpx

SUPPORTED_EXT = {".pdf", ".txt", ".md", ".docx"}


@dataclass
class Chunk:
    """A single text chunk with provenance."""
    text: str
    source: str
    chunk_no: int


def read_pdf(path: Path) -> str:
    """Extract text from a PDF using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def read_docx(path: Path) -> str:
    """Extract text from a .docx file."""
    import docx
    d = docx.Document(str(path))
    return "\n".join(p.text for p in d.paragraphs)


def read_text(path: Path) -> str:
    """Read .txt / .md as UTF-8 (best effort)."""
    return path.read_text(encoding="utf-8", errors="ignore")


def load_document(path: Path) -> str:
    """Dispatch to the right reader based on extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    if ext in {".txt", ".md"}:
        return read_text(path)
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, size: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Whitespace-token chunker with a fixed-size sliding window.

    We approximate "tokens" as whitespace-separated words. This is intentionally
    cheap and deterministic; the embedding model will re-tokenize internally.
    """
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


def file_signature(path: Path) -> str:
    """Stable signature of a file's path + size + mtime (cheap dedupe key)."""
    stat = path.stat()
    raw = f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()


def load_metadata(path: Path) -> dict:
    """Load the metadata sidecar or return an empty skeleton."""
    if path.exists():
        return json.loads(path.read_text())
    return {"files": {}, "chunks": []}


def save_metadata(meta: dict, path: Path) -> None:
    """Persist the metadata sidecar."""
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def load_or_create_index(index_path: Path) -> tuple[TurboQuantIndex, bool]:
    """Load an existing TurboQuantIndex or create a fresh one. Returns (index, existed)."""
    if index_path.exists():
        try:
            return TurboQuantIndex.load(str(index_path)), True
        except Exception as e:
            print(f"  [warn] failed to load existing index ({e}); starting fresh")
    return TurboQuantIndex(dim=EMBED_DIM, bit_width=BIT_WIDTH), False


def iter_documents(source: Path) -> Iterable[Path]:
    """Yield every supported file under `source` recursively."""
    for p in sorted(source.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            yield p


def ingest(source: Path, index_path: Path, metadata_path: Path, *, build_graph: bool = True) -> dict:
    """Run the full ingestion pass.

    For every new chunk we:
      1. embed it into the TurboVec index
      2. (if build_graph) extract entities + relations via DeepSeek and merge
         them into the persisted knowledge graph
    After all files are processed we re-run community detection + summarisation
    on the full graph (cheap relative to the per-chunk extraction cost).
    """
    meta = load_metadata(metadata_path)
    index, existed = load_or_create_index(index_path)

    if not existed:
        meta = {"files": {}, "chunks": []}

    model = SentenceTransformer(EMBED_MODEL_NAME)
    kg = KnowledgeGraph.load(GRAPH_PATH) if build_graph else None

    total_added = 0
    files_processed = 0
    files_skipped = 0
    triples_added = 0

    extract_client = httpx.Client(timeout=60) if build_graph else None
    try:
        for path in iter_documents(source):
            sig = file_signature(path)
            key = str(path.resolve())
            if meta["files"].get(key, {}).get("signature") == sig:
                files_skipped += 1
                continue

            try:
                text = load_document(path)
            except Exception as e:
                print(f"  [skip] {path.name}: {e}")
                continue

            chunks = chunk_text(text)
            if not chunks:
                print(f"  [skip] {path.name}: empty after chunking")
                continue

            print(f"Ingesting {len(chunks)} chunks from {path.name}...")
            vectors = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
            vectors = np.asarray(vectors, dtype=np.float32)
            index.add(vectors)

            start_idx = len(meta["chunks"])
            for i, ch in enumerate(chunks):
                chunk_id = start_idx + i
                meta["chunks"].append({"source": path.name, "path": key, "chunk_no": i, "text": ch})
                if build_graph and kg is not None:
                    try:
                        ex = extract_chunk(ch, client=extract_client)
                    except Exception as e:
                        print(f"  [extract:warn] chunk {chunk_id}: {e}")
                        continue
                    kg.ingest_extraction(ex, chunk_id)
                    triples_added += len(ex.entities) + len(ex.relations)

            meta["files"][key] = {"signature": sig, "chunks": len(chunks), "name": path.name}
            total_added += len(chunks)
            files_processed += 1
    finally:
        if extract_client is not None:
            extract_client.close()

    if total_added > 0:
        index.write(str(index_path))
    save_metadata(meta, metadata_path)

    communities_count = 0
    if build_graph and kg is not None and total_added > 0:
        kg.save(GRAPH_PATH)
        if kg.g.number_of_nodes() > 0:
            print("Detecting communities and writing summaries...")
            communities = build_communities(kg)
            save_communities(communities, COMMUNITIES_PATH)
            communities_count = len(communities)

    summary = {
        "files_processed": files_processed,
        "files_skipped": files_skipped,
        "chunks_added": total_added,
        "total_chunks": len(meta["chunks"]),
        "triples_added": triples_added,
        "communities": communities_count,
        "graph_nodes": kg.g.number_of_nodes() if kg else 0,
        "graph_edges": kg.g.number_of_edges() if kg else 0,
    }
    print(f"Done. {summary}")
    return summary


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Ingest documents into the TurboVec RAG index.")
    ap.add_argument("--source", type=Path, default=RAW_DIR, help="Directory with raw documents")
    ap.add_argument("--index", type=Path, default=INDEX_PATH, help="Path to write the TurboVec index")
    ap.add_argument("--metadata", type=Path, default=METADATA_PATH, help="Path to metadata JSON")
    ap.add_argument("--no-graph", action="store_true", help="Skip Graph-RAG extraction (vector-only ingest)")
    args = ap.parse_args()
    ingest(args.source, args.index, args.metadata, build_graph=not args.no_graph)
    return 0


if __name__ == "__main__":
    sys.exit(main())
