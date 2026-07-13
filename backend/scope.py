"""Per-user / per-school document scoping.

The TurboVec index, metadata sidecar, and knowledge graph are process-global.
Before this module existed, every signed-in user could retrieve every other
user's uploads — fine for a single-school box, a data leak on the public
multi-tenant deployment.

Design: ownership lives in a sidecar (data/ownership.json), keyed by the
file's absolute path (the same key ingest.py uses in metadata["files"]).
Retrieval callers compute the caller's allowed-path set once per request and
pass it down; the retriever hard-filters chunks by chunk["path"], and graph
context is filtered to nodes whose chunks live in allowed files.

Sharing rules:
  - The uploader always sees their own files.
  - If the uploader is a teacher or admin of a school, the file is shared
    with that school: every member of the school sees it.
  - Files ingested before scoping existed have no ownership record ("legacy").
    They are hidden by default; set VAAANI_LEGACY_DOCS_SHARED=1 on a
    single-school deployment where the shared corpus is intentional, or
    assign owners with scripts/assign_owner.py.
  - VAAANI_SCOPE_DISABLED=1 turns scoping off entirely (old behaviour).
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from config import DATA_DIR

OWNERSHIP_PATH = DATA_DIR / "ownership.json"
_LOCK = threading.Lock()

SCOPE_DISABLED = os.environ.get("VAAANI_SCOPE_DISABLED", "0") == "1"
LEGACY_SHARED = os.environ.get("VAAANI_LEGACY_DOCS_SHARED", "0") == "1"

# Sentinel path that matches no chunk — used to force empty retrieval when a
# user's allowed set is empty (an empty set would read as "no filter").
NOTHING = "__no_documents__"


def _load() -> dict:
    if OWNERSHIP_PATH.exists():
        try:
            return json.loads(OWNERSHIP_PATH.read_text())
        except Exception:
            return {}
    return {}


def record_ownership(file_key: str, user_id: int, school_ids: list[int] | None = None) -> None:
    """Idempotently register `user_id` (and optional schools) as owners of a file."""
    with _LOCK:
        data = _load()
        rec = data.get(file_key) or {"owners": [], "school_ids": []}
        if user_id not in rec["owners"]:
            rec["owners"].append(user_id)
        for sid in school_ids or []:
            if sid not in rec["school_ids"]:
                rec["school_ids"].append(sid)
        data[file_key] = rec
        OWNERSHIP_PATH.write_text(json.dumps(data, indent=1))


def record_library_ownership(file_key: str) -> None:
    """Mark a file as Vaaani Core Library content — visible to EVERY learner.

    The Library is the curated curriculum a learner explores without uploading
    anything (the B2C young-learner face). It rides the same scoping machinery
    as private/school docs; it's simply readable by all, so a brand-new learner
    opens a populated universe instead of an empty map.
    """
    with _LOCK:
        data = _load()
        rec = data.get(file_key) or {"owners": [], "school_ids": []}
        rec["library"] = True
        data[file_key] = rec
        OWNERSHIP_PATH.write_text(json.dumps(data, indent=1))


def sharing_school_ids(user: dict | None) -> list[int]:
    """School ids a user's uploads are shared with: schools where they are
    teacher or admin. Students' uploads stay personal."""
    if not user:
        return []
    try:
        from auth.school import list_schools_for_user
        return [
            s["id"] for s in list_schools_for_user(user["id"])
            if s.get("role") in ("teacher", "admin")
        ]
    except Exception:
        return []


def member_school_ids(user: dict | None) -> set[int]:
    """All schools the user belongs to, in any role (for read access)."""
    if not user:
        return set()
    try:
        from auth.school import list_schools_for_user
        return {s["id"] for s in list_schools_for_user(user["id"])}
    except Exception:
        return set()


def allowed_paths_for(user: dict | None, metadata_files: dict) -> set[str] | None:
    """Compute the set of file keys `user` may read, or None for "no filter".

    None (unrestricted) only when scoping is disabled. Otherwise always a
    non-empty set — NOTHING is added so an empty allowance still filters.
    """
    if SCOPE_DISABLED:
        return None
    ownership = _load()
    allowed: set[str] = {NOTHING}
    uid = user["id"] if user else None
    schools = member_school_ids(user)
    for key in metadata_files:
        rec = ownership.get(key)
        if rec is None:
            if LEGACY_SHARED:
                allowed.add(key)
            continue
        if rec.get("library"):
            allowed.add(key)                      # Core Library — visible to all
        elif uid is not None and uid in rec.get("owners", []):
            allowed.add(key)
        elif schools and schools & set(rec.get("school_ids", [])):
            allowed.add(key)
    # Legacy chunks that predate the "path" field can only ever match ""
    if LEGACY_SHARED:
        allowed.add("")
    return allowed
