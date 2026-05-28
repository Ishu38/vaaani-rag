"""FastAPI app exposing /chat, /ingest, /status and the static frontend."""
from __future__ import annotations

import shutil
import sqlite3 as _sqlite3
from pathlib import Path

from fastapi import Cookie, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from adaptive import service as learn_service
from adaptive.routes import router as learning_router
from audio import (
    available_voices,
    cache_path_for,
    list_narratable_docs,
    narrate_doc,
    podcast_doc,
)
from feynman import diff_explanation, list_topics
from messenger.routes import router as messenger_router
from youtube.routes import router as youtube_router
from adaptive import service as learn_service
from auth import service as auth_service
from auth.routes import router as auth_router
from auth.school import (
    build_guardrail_prompt,
    build_universal_guardrail_prompt,
    check_guardrail_violation,
    get_student_guardrails,
    log_guardrail_event,
)
from auth.security import decode_session
from hermes import corrector as hermes_corrector, store as hermes_store
from hermes.routes import router as hermes_router
from cognitive.routes import router as cognitive_router
from simulation.routes import router as simulation_router

from config import (
    DATA_DIR,
    INDEX_PATH,
    MAX_UPLOAD_BYTES,
    METADATA_PATH,
    RAW_DIR,
    ROOT,
    STRUCTURED_TRIGGERS,
    TOP_K,
)
from ingest import ingest, ingest_vectors, ingest_graph_deferred, SUPPORTED_EXT
from diagram import extract_and_render_all as render_diagrams
from intent import classify, graph_mode, wants_structured_output
from llm import (
    LLMResponse,
    build_graph_block,
    build_prompt,
    call_deepseek,
    citation_fidelity,
    maybe_parse_structured,
)
from memory import (
    format_memory_block,
    load_memory,
    record_query,
    top_relevant_facts,
)
from retriever import Retriever

app = FastAPI(title="Local RAG Assistant", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(learning_router)
app.include_router(hermes_router)
app.include_router(messenger_router)
app.include_router(youtube_router)
app.include_router(cognitive_router)
app.include_router(simulation_router)
hermes_store.init_hermes_db()

# Initialize async ingest job tracker (SQLite-backed, survives restarts)
_JOBS_DB_PATH = DATA_DIR / "jobs.db"

def _init_jobs_db() -> None:
    """Create the jobs table if it doesn't exist. Idempotent."""
    _JOBS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = _sqlite3.connect(str(_JOBS_DB_PATH))
    db.execute("""
        CREATE TABLE IF NOT EXISTS ingest_jobs (
            job_id      TEXT PRIMARY KEY,
            status      TEXT NOT NULL DEFAULT 'queued',
            phase       TEXT DEFAULT 'queued',
            filename    TEXT DEFAULT '',
            size_bytes  INTEGER DEFAULT 0,
            chunks_added   INTEGER DEFAULT 0,
            total_chunks   INTEGER DEFAULT 0,
            triples_added  INTEGER DEFAULT 0,
            communities    INTEGER DEFAULT 0,
            error       TEXT DEFAULT '',
            queued_at   REAL DEFAULT 0,
            started_at  REAL DEFAULT 0,
            completed_at REAL DEFAULT 0,
            extracted      INTEGER DEFAULT 0,
            extract_total  INTEGER DEFAULT 0
        )
    """)
    db.commit()
    # Best-effort column-add for older DBs (idempotent — ignore "duplicate column"
    # errors). Keeps existing deploys upgradable without a manual migration.
    for col in ("extracted INTEGER DEFAULT 0", "extract_total INTEGER DEFAULT 0"):
        try:
            db.execute(f"ALTER TABLE ingest_jobs ADD COLUMN {col}")
        except _sqlite3.OperationalError:
            pass
    db.commit()
    db.execute(
        "UPDATE ingest_jobs SET status='abandoned', phase='lost_on_restart' "
        "WHERE status IN ('queued', 'running')"
    )
    db.commit()
    db.close()

_init_jobs_db()

# Initialize cognitive and simulation databases
try:
    from cognitive.store import init_db as init_cognitive_db
    init_cognitive_db()
except Exception:
    pass

try:
    from simulation.store import init_db as init_simulation_db
    init_simulation_db()
except Exception:
    pass

retriever = Retriever()
FRONTEND_INDEX = ROOT / "frontend" / "index.html"
FRONTEND_GRAPH = ROOT / "frontend" / "graph.html"
SITE_DIR = ROOT / "site"
SITE_INDEX = SITE_DIR / "index.html"
# Rendered plots written by backend/plot.py. Lives under data/ so the systemd
# unit's ReadWritePaths (data + data/raw) already covers it.
FIGURES_DIR = DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

if SITE_DIR.exists():
    # Serve /site/style.css, /site/main.js, /site/assets/... as static.
    app.mount("/site", StaticFiles(directory=str(SITE_DIR)), name="site")

# Serve LLM-generated diagram PNGs. Filenames are uuid4 hex (no traversal risk).
app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")


class ChatRequest(BaseModel):
    """Body for POST /chat."""
    query: str = Field(..., min_length=1)
    conversation_history: list[dict] = Field(default_factory=list)
    remember: str | None = Field(
        default=None,
        description="Optional fact to persist into long-term memory.",
    )
    socratic: bool = Field(
        default=False,
        description="When true, assistant teaches by asking leading questions instead of answering directly.",
    )
    source_filter: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of source filenames to scope retrieval to. When "
            "provided, only chunks whose source is in this list are eligible "
            "for vector hits + graph expansion. None or [] = no filter "
            "(search the whole corpus, current behaviour)."
        ),
    )


class HermesCorrection(BaseModel):
    """One advisory adjustment the Hermes corrector applied to this turn."""
    name: str
    reason: str


class PlotFigure(BaseModel):
    """One LLM-requested figure rendered by backend/plot.py. The SPA splits
    the answer text on [[FIG:id]] sentinels and inlines an <img> for each."""
    id: str
    url: str
    caption: str
    expr: str


class ChatResponse(BaseModel):
    """Body returned from POST /chat."""
    answer: str
    sources: list[dict]
    tokens: int
    intent: str
    graph_mode: str | None = None
    entities: list[str] = Field(default_factory=list)
    topic_refs: list[dict] = Field(default_factory=list)
    communities: list[dict] = Field(default_factory=list)
    structured: dict | None = None
    fidelity_warnings: list[str] = Field(default_factory=list)
    memory_used: list[str] = Field(default_factory=list)
    weak_spots: list[dict] = Field(default_factory=list)
    user_signed_in: bool = False
    hermes_corrections: list[HermesCorrection] = Field(default_factory=list)
    figures: list[PlotFigure] = Field(default_factory=list)
    guardrail_active: bool = False
    guardrail_violations: list[dict] = Field(default_factory=list)


@app.get("/")
def root() -> FileResponse:
    """Serve the marketing landing page. Falls back to chat if the site isn't built."""
    if SITE_INDEX.exists():
        return FileResponse(SITE_INDEX)
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    raise HTTPException(404, "Neither site/index.html nor frontend/index.html found")


@app.get("/app")
def chat_app(vaaani_session: str | None = Cookie(default=None, alias="vaaani_session")):
    """Serve the chat assistant SPA. Requires authentication — anonymous
    visitors are redirected to /login. Vaaani's chat is not a public free
    surface; the public surface is /dashboard's preview."""
    from fastapi.responses import RedirectResponse
    user = _resolve_user(vaaani_session)
    if not user:
        return RedirectResponse(url="/login?next=/app", status_code=302)
    if not FRONTEND_INDEX.exists():
        raise HTTPException(404, "frontend/index.html not found")
    return FileResponse(FRONTEND_INDEX)


def _serve_site(name: str) -> FileResponse:
    """Helper: serve a static file from site/, with a 404 if missing."""
    p = SITE_DIR / name
    if not p.exists():
        raise HTTPException(404, f"site/{name} not found")
    return FileResponse(p)


@app.get("/about")
def about_page() -> FileResponse:
    """Serve the About page (builder bio + three lenses)."""
    return _serve_site("about.html")


@app.get("/contact")
def contact_page() -> FileResponse:
    """Serve the Contact page (correspondence address + map embed)."""
    return _serve_site("contact.html")


@app.get("/integrations")
def integrations_page() -> FileResponse:
    return _serve_site("integrations.html")


@app.get("/pricing")
def pricing_page() -> FileResponse:
    """Serve the Pricing page with three tiers."""
    return _serve_site("pricing.html")


@app.get("/signup")
def signup_page() -> FileResponse:
    """Serve the sign-up form."""
    return _serve_site("signup.html")


@app.get("/login")
def login_page() -> FileResponse:
    """Serve the sign-in form."""
    return _serve_site("login.html")


@app.get("/verify")
def verify_page() -> FileResponse:
    """Serve the verification landing page (handles ?status=ok|invalid)."""
    return _serve_site("verify.html")


@app.get("/account")
def account_page() -> FileResponse:
    """Serve the student dashboard (mastery, review queue, profile)."""
    return _serve_site("account.html")


@app.get("/parental-consent")
def parental_consent_page() -> FileResponse:
    """Parent landing page for the DPDP §9 consent magic link. The page reads
    ?token=... from the URL, fetches /auth/consent/lookup/{token}, renders the
    consent text + child summary, and posts /auth/consent/confirm on submit."""
    return _serve_site("parental-consent.html")


@app.get("/privacy")
def privacy_page() -> FileResponse:
    """Serve the privacy notice (template — lawyer must review before
    relying on it for DPDP §5 compliance)."""
    return _serve_site("privacy.html")


@app.get("/cognitive")
def cognitive_page() -> FileResponse:
    """Cognitive X-Ray — detect thinking flaws in a learner's reasoning.
    Pairs with the cognitive_router (/cognitive/analyze, /fingerprint, etc.)
    for the API surface. Page-route added 2026-05-28 so the chat nav can
    link to it directly with /cognitive."""
    return _serve_site("cognitive.html")


@app.get("/simulation")
def simulation_page() -> FileResponse:
    """Pressure Simulation — exam-style time-boxed practice. Pairs with
    the simulation_router (/simulation/start, /answer, /report, etc.)
    for the API surface. Page-route added 2026-05-28 so the chat nav
    can link to it directly with /simulation."""
    return _serve_site("simulation.html")


@app.get("/dashboard")
def school_dashboard_page() -> FileResponse:
    """Serve the teacher/school-admin dashboard."""
    return _serve_site("dashboard.html")


@app.get("/graph-view")
def graph_view(vaaani_session: str | None = Cookie(default=None, alias="vaaani_session")):
    """Serve the interactive knowledge-graph visualisation page.

    Auth-gated: the graph reveals every entity in the corpus, so anonymous
    visitors get bounced to /login instead of seeing another user's data.
    """
    from fastapi.responses import RedirectResponse
    if not _resolve_user(vaaani_session):
        return RedirectResponse(url="/login?next=/graph-view", status_code=302)
    if not FRONTEND_GRAPH.exists():
        raise HTTPException(404, "frontend/graph.html not found")
    return FileResponse(FRONTEND_GRAPH)


@app.get("/cognitive")
def cognitive_page(vaaani_session: str | None = Cookie(default=None, alias="vaaani_session")):
    """Serve the Cognitive X-Ray fingerprint dashboard."""
    from fastapi.responses import RedirectResponse
    if not _resolve_user(vaaani_session):
        return RedirectResponse(url="/login?next=/cognitive", status_code=302)
    return _serve_site("cognitive.html")


@app.get("/simulation")
def simulation_page(vaaani_session: str | None = Cookie(default=None, alias="vaaani_session")):
    """Serve the Exam Pressure Simulation interface."""
    from fastapi.responses import RedirectResponse
    if not _resolve_user(vaaani_session):
        return RedirectResponse(url="/login?next=/simulation", status_code=302)
    return _serve_site("simulation.html")


@app.get("/status")
def status(vaaani_session: str | None = Cookie(default=None, alias="vaaani_session")) -> dict:
    """Return index size, chunk count, and indexed documents.

    Anonymous callers get zeros — the corpus counts (docs/chunks/entities)
    are private to signed-in users and must not leak via the public pill.
    """
    if not _resolve_user(vaaani_session):
        # Same schema as retriever.status() but zeroed — keeps the SPA's
        # refreshStatus() happy without exposing corpus counts to anon callers.
        return {
            "total_chunks": 0,
            "index_size_mb": 0.0,
            "documents_indexed": [],
            "embedding_dim": 0,
            "bit_width": 0,
            "graph_nodes": 0,
            "graph_edges": 0,
            "communities_count": 0,
            "memory_facts": 0,
            "recent_queries": [],
        }
    s = retriever.status()
    mem = load_memory()
    s["memory_facts"] = len(mem.get("facts", []))
    s["recent_queries"] = mem.get("recent_queries", [])[-5:]
    return s


class NarrateRequest(BaseModel):
    doc_name: str
    voice: str | None = None
    # "narration" → single-voice readback; "podcast" → 2-host dialogue
    mode: str = "narration"


@app.get("/audio/library")
def audio_library() -> dict:
    """Ingested docs eligible for narration plus the voice list."""
    return {
        "docs": list_narratable_docs(),
        "voices": available_voices(),
    }


@app.post("/audio/narrate")
def audio_narrate(req: NarrateRequest) -> dict:
    """Synthesize an MP3 for an ingested document. Idempotent via SHA1 cache."""
    mode = (req.mode or "narration").lower()
    try:
        if mode == "podcast":
            result = podcast_doc(req.doc_name)
        elif mode == "narration":
            result = narrate_doc(req.doc_name, voice=req.voice)
        else:
            raise HTTPException(400, f"unknown mode '{req.mode}'")
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    except KeyError as e:
        raise HTTPException(404, str(e))
    except (RuntimeError, ValueError) as e:
        raise HTTPException(500, f"audio generation failed: {e}")
    return {
        "cache_hash": result.cache_hash,
        "url": f"/audio/file/{result.cache_hash}.mp3",
        "duration_s": round(result.duration_s, 2),
        "voice": result.voice,
        "doc_name": result.doc_name,
        "cached": result.cached,
        "mode": mode,
    }


class FeynmanRequest(BaseModel):
    topic_id: str
    explanation: str
    k: int = 2


@app.get("/feynman/topics")
def feynman_topics() -> dict:
    """Topics worth explaining back: well-connected graph nodes, ranked
    by degree desc."""
    return {"topics": list_topics()}


@app.post("/feynman/diff")
def feynman_diff(req: FeynmanRequest) -> dict:
    """Run the explain-it-back diff against the corpus subgraph for the
    chosen topic. Returns structured node/edge coverage."""
    if len(req.explanation.strip()) < 20:
        raise HTTPException(400, "explanation is too short — write at least a few sentences")
    if not (1 <= req.k <= 3):
        raise HTTPException(400, "k must be between 1 and 3")
    try:
        result = diff_explanation(req.explanation, req.topic_id, k=req.k)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, f"diff failed: {e}")
    return result.to_json()


@app.get("/audio/file/{filename}")
def audio_file(filename: str) -> FileResponse:
    # Only 16-hex-char SHA1 prefixes plus .mp3; reject anything else to
    # block path traversal into data/audio/.
    if not filename.endswith(".mp3"):
        raise HTTPException(400, "expected .mp3")
    stem = filename[:-4]
    if len(stem) != 16 or not all(c in "0123456789abcdef" for c in stem):
        raise HTTPException(400, "invalid hash")
    path = cache_path_for(stem)
    if not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="audio/mpeg")


@app.post("/ingest")
def ingest_endpoint(
    file: UploadFile = File(...),
    vaaani_session: str | None = Cookie(default=None, alias="vaaani_session"),
) -> dict:
    """Accept an uploaded document, write it to data/raw/, and re-ingest.

    Synchronous — runs to completion in-request. Best for small files (<5 MB).
    For larger uploads use /ingest/async + /ingest/status/{job_id} to avoid
    Cloudflare's 100s edge timeout. Frontend uses /ingest/async by default."""
    _resolve_processing_user(vaaani_session)
    name = Path(file.filename or "upload.bin").name
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {sorted(SUPPORTED_EXT)}")
    dest = RAW_DIR / name
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    size_bytes = dest.stat().st_size
    if size_bytes > MAX_UPLOAD_BYTES:
        dest.unlink(missing_ok=True)
        raise HTTPException(413, f"File too large: {size_bytes/1e6:.1f} MB. Max: {MAX_UPLOAD_BYTES/1e6:.0f} MB")
    summary = ingest(RAW_DIR, INDEX_PATH, METADATA_PATH)
    retriever.reload()
    return {
        "status": "ok",
        "filename": name,
        "chunks_added": summary["chunks_added"],
        "total_chunks": summary["total_chunks"],
    }


# ---- Async ingest -----------------------------------------------------------
# SQLite-backed job tracker. Survives server restarts — jobs persist in
# data/jobs.db. On startup, any queued/running jobs from a previous process
# are marked as "abandoned".

import threading as _threading
import uuid as _uuid
from concurrent.futures import ThreadPoolExecutor as _Pool

_INGEST_JOBS: dict[str, dict] = {}
_INGEST_LOCK = _threading.Lock()
_INGEST_POOL = _Pool(max_workers=1, thread_name_prefix="ingest")


def _save_job(job: dict) -> None:
    """Persist a job record to SQLite (upsert)."""
    db = _sqlite3.connect(str(_JOBS_DB_PATH))
    db.execute(
        """INSERT OR REPLACE INTO ingest_jobs
           (job_id, status, phase, filename, size_bytes, chunks_added,
            total_chunks, triples_added, communities, error,
            queued_at, started_at, completed_at, extracted, extract_total)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            job.get("job_id", ""),
            job.get("status", "queued"),
            job.get("phase", "queued"),
            job.get("filename", ""),
            job.get("size_bytes", 0),
            job.get("chunks_added", 0),
            job.get("total_chunks", 0),
            job.get("triples_added", 0),
            job.get("communities", 0),
            job.get("error", ""),
            job.get("queued_at", 0),
            job.get("started_at", 0),
            job.get("completed_at", 0),
            job.get("extracted", 0),
            job.get("extract_total", 0),
        ),
    )
    db.commit()
    db.close()


def _load_job(job_id: str) -> dict | None:
    """Read a single job from SQLite, or None if missing."""
    db = _sqlite3.connect(str(_JOBS_DB_PATH))
    row = db.execute(
        "SELECT * FROM ingest_jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    db.close()
    if not row:
        return None
    cols = ["job_id", "status", "phase", "filename", "size_bytes",
            "chunks_added", "total_chunks", "triples_added", "communities",
            "error", "queued_at", "started_at", "completed_at",
            "extracted", "extract_total"]
    return dict(zip(cols, row))


def _update_job(job_id: str, **patch) -> None:
    with _INGEST_LOCK:
        if job_id in _INGEST_JOBS:
            _INGEST_JOBS[job_id].update(patch)
            _save_job(_INGEST_JOBS[job_id])


def _run_ingest_job(job_id: str, filename: str) -> None:
    """Two-phase ingest: vectors (instant) → retriever reload → graph (background).

    Phase 1 makes the file searchable in ~1s. Phase 2 builds the knowledge
    graph and communities, which may take minutes — but the file is already
    answering questions via vector search while that happens.
    """
    import time as _time
    _update_job(job_id, status="running", phase="vectors", started_at=_time.time())
    try:
        # ── Phase 1: vectors only (fast) ──
        _update_job(job_id, phase="embedding")
        summary = ingest_vectors(RAW_DIR, INDEX_PATH, METADATA_PATH)
        _update_job(job_id, phase="reloading")
        retriever.reload()
        _update_job(
            job_id,
            status="graph_pending",
            phase="vectors_done",
            chunks_added=summary.get("chunks_added", 0),
            total_chunks=summary.get("total_chunks", 0),
            filename=filename,
        )

        # ── Phase 2: graph extraction (background) ──
        if summary.get("chunks_added", 0) > 0:
            _update_job(job_id, phase="extracting", extracted=0, extract_total=0)

            # Live progress: fires on every batch completion. We rate-limit
            # SQLite writes to once per ~2 chunks of progress to keep the
            # DB calm during big ingests.
            _last_persist = {"done": -10}
            def _on_progress(done: int, total: int) -> None:
                if done - _last_persist["done"] >= 2 or done == total:
                    _update_job(job_id, extracted=done, extract_total=total)
                    _last_persist["done"] = done

            graph_result = ingest_graph_deferred(progress_cb=_on_progress)
            _update_job(job_id, phase="reloading")
            retriever.reload()  # reload again so communities are visible
            _update_job(
                job_id,
                status="complete",
                phase="done",
                completed_at=_time.time(),
                triples_added=graph_result.get("triples_added", 0),
                communities=graph_result.get("communities", 0),
                filename=filename,
            )
        else:
            _update_job(job_id, status="complete", phase="done",
                        completed_at=_time.time(), filename=filename)
    except Exception as e:
        _update_job(
            job_id,
            status="failed",
            phase="error",
            completed_at=_time.time(),
            error=str(e)[:500],
        )


@app.post("/ingest/async", status_code=202)
def ingest_async(
    file: UploadFile = File(...),
    vaaani_session: str | None = Cookie(default=None, alias="vaaani_session"),
) -> dict:
    """Queue a document ingest job. Returns 202 + job_id immediately; client
    polls /ingest/status/{job_id} for progress. Avoids Cloudflare's 100s
    edge-timeout that kills sync ingest of larger PDFs."""
    _resolve_processing_user(vaaani_session)
    name = Path(file.filename or "upload.bin").name
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {sorted(SUPPORTED_EXT)}")
    dest = RAW_DIR / name
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    size_bytes = dest.stat().st_size
    if size_bytes > MAX_UPLOAD_BYTES:
        dest.unlink(missing_ok=True)
        raise HTTPException(413, f"File too large: {size_bytes/1e6:.1f} MB. Max: {MAX_UPLOAD_BYTES/1e6:.0f} MB")

    job_id = _uuid.uuid4().hex[:16]
    with _INGEST_LOCK:
        _INGEST_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "phase": "queued",
            "filename": name,
            "size_bytes": size_bytes,
            "queued_at": __import__("time").time(),
        }
        # Bound the job dict — keep only last 50, drop oldest finished jobs.
        if len(_INGEST_JOBS) > 50:
            done = sorted(
                ((k, v) for k, v in _INGEST_JOBS.items() if v["status"] in ("complete", "failed")),
                key=lambda kv: kv[1].get("completed_at", 0),
            )
            for k, _ in done[: max(0, len(_INGEST_JOBS) - 50)]:
                _INGEST_JOBS.pop(k, None)
    _INGEST_POOL.submit(_run_ingest_job, job_id, name)
    return {"job_id": job_id, "status": "queued", "filename": name, "size_bytes": size_bytes}


@app.get("/ingest/status/{job_id}")
def ingest_status(job_id: str) -> dict:
    """Poll for the status of an async ingest job. Falls back to SQLite
    so jobs survive server restarts."""
    with _INGEST_LOCK:
        job = _INGEST_JOBS.get(job_id)
    if not job:
        job = _load_job(job_id)
    if not job:
        raise HTTPException(404, f"Unknown job_id: {job_id}")
    return dict(job)


def _resolve_user(cookie: str | None) -> dict | None:
    """Look up the current user from the session cookie, or None if unauth."""
    payload = decode_session(cookie or "")
    if not payload:
        return None
    try:
        return auth_service.get_user_by_id(int(payload["sub"]))
    except (KeyError, ValueError):
        return None


def _resolve_processing_user(cookie: str | None) -> dict:
    """Require an authenticated user AND a DPDP-compliant processing state.

    Used by gated endpoints (/chat, /chat/stream, /ingest, /ingest/async).
    Raises 401 for anonymous, 403 with a machine-readable reason for users
    whose consent state forbids processing. The reason codes are stable:
      consent_required   — under-18, parent has not yet confirmed
      consent_withdrawn  — parent has revoked; account is locked
      account_deleted    — soft-deleted via /auth/data-delete
    """
    user = _resolve_user(cookie)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to use Vaaani.")
    from auth import dpdp as _dpdp
    allowed, reason = _dpdp.allow_processing(user)
    if not allowed:
        _dpdp.audit(user["id"], "access_blocked", f"reason={reason}")
        raise HTTPException(status_code=403, detail=reason or "Processing not permitted.")
    return user


def _run_intent(
    query: str,
    structured: bool,
    *,
    socratic: bool = False,
    user: dict | None = None,
    guardrail_prompt: str = "",
    source_filter: list[str] | None = None,
) -> tuple[LLMResponse, dict]:
    """Route a query through intent → graph-aware retrieval → LLM.

    Returns (LLMResponse, retrieval_payload) where retrieval_payload exposes
    the chunks/entities/communities/edges used so the API layer can surface
    them in the response body.
    """
    intent = classify(query)
    g_mode: str | None = None
    retrieval: dict = {"chunks": [], "entities": [], "communities": [], "edges": []}

    proposed_g_mode = graph_mode(query) if intent == "knowledge" else None

    # ---- Hermes pre-flight: consult past traces, get correction plan ----
    query_vec = retriever.embed([query])[0]
    user_id = user["id"] if user else None
    try:
        hermes_plan = hermes_corrector.plan(
            query_vec,
            user_id=user_id,
            intent=intent,
            proposed_graph_mode=proposed_g_mode,
        )
    except Exception:
        hermes_plan = hermes_corrector.CorrectionPlan([], 0, 0.0, 0.0)
    correction_names = set(hermes_plan.names)

    # Apply: upgrade local→global if Hermes asks for it.
    if intent == "knowledge":
        g_mode = "global" if "upgrade_graph_global" in correction_names else proposed_g_mode

    # Apply: broaden retrieval (raise top_k) when neighbours were chunk-starved.
    effective_top_k = TOP_K * 2 if "broaden_retrieval" in correction_names else TOP_K

    sf_set: set[str] | None = set(source_filter) if source_filter else None

    if intent == "knowledge":
        retrieval = (
            retriever.global_graph_search(query)
            if g_mode == "global"
            else retriever.local_graph_search(query, k=effective_top_k, source_filter=sf_set)
        )
    elif intent == "task":
        retrieval["chunks"] = retriever.search(query, k=effective_top_k, source_filter=sf_set)

    chunks = retrieval["chunks"]
    facts = top_relevant_facts(query, retriever.embed)
    memory_block = format_memory_block(facts)

    # Build canonical topic refs (key + display) and look up weak-spots for the user.
    entity_displays = retrieval.get("entities", []) or []
    topic_refs: list[dict] = []
    weak: list[dict] = []
    if entity_displays:
        for d in entity_displays:
            key = learn_service.normalize_topic(d)
            if key:
                topic_refs.append({"topic": key, "display": d})
        if user and topic_refs:
            try:
                weak = learn_service.weak_spots(user["id"], [t["topic"] for t in topic_refs])
            except Exception:
                weak = []

    graph_block = (
        build_graph_block(retrieval["entities"], retrieval["edges"], retrieval["communities"])
        if intent == "knowledge"
        else ""
    )
    if weak and socratic:
        bullets = "\n".join(f"- {w['display']} (student-rated mastery {w['mastery']:.1f}/5)" for w in weak)
        graph_block = (graph_block + "\n\n" if graph_block else "") + (
            "STUDENT WEAK SPOTS (from prior ratings) — bias your Socratic questions toward these:\n"
            + bullets
        )
    # Apply: strict-grounding directive when neighbours produced unsupported claims.
    extra_system = (
        hermes_corrector.STRICT_GROUNDING_DIRECTIVE
        if "strict_grounding" in correction_names
        else ""
    )
    messages = build_prompt(
        query, chunks, memory_block, intent, structured,
        graph_mode=g_mode, graph_block=graph_block, socratic=socratic,
        extra_system=extra_system,
        guardrail_prompt=guardrail_prompt,
    )
    resp = call_deepseek(messages, stream=False, json_mode=structured)

    choice = resp.get("choices", [{}])[0]
    answer = choice.get("message", {}).get("content", "")
    tokens = resp.get("usage", {}).get("total_tokens", 0)

    structured_payload = maybe_parse_structured(answer) if structured else None
    # Socratic answers are questions, not claims — skip fidelity entirely.
    warnings = (
        citation_fidelity(answer, chunks)
        if intent == "knowledge" and g_mode == "local" and not socratic
        else []
    )

    llm_resp = LLMResponse(
        answer=answer,
        sources_used=[c.get("source", "") for c in chunks],
        tokens_used=tokens,
        structured=structured_payload,
        fidelity_warnings=warnings,
        intent=intent,
    )

    # Touch student skills (no rating) so the dashboard shows topics they've engaged with.
    if user and topic_refs:
        subject = learn_service.classify_subject(query, " ".join(t["display"] for t in topic_refs))
        for t in topic_refs:
            try:
                learn_service.upsert_skill(user["id"], t["topic"], t["display"], subject)
            except Exception:
                pass

    # ---- Hermes post-flight: record this turn for future self-correction ----
    try:
        hermes_store.log_trace(
            user_id=user_id,
            query=query,
            embedding=query_vec,
            intent=intent,
            graph_mode=g_mode,
            num_chunks=len(chunks),
            fidelity_warnings=len(warnings),
            tokens=tokens,
            corrections_applied=hermes_plan.names,
        )
    except Exception:
        pass

    return llm_resp, {
        **retrieval,
        "graph_mode": g_mode,
        "topic_refs": topic_refs,
        "weak_spots": weak,
        "hermes_corrections": [
            {"name": c.name, "reason": c.reason} for c in hermes_plan.corrections
        ],
    }


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    vaaani_session: str | None = Cookie(default=None, alias="vaaani_session"),
) -> ChatResponse:
    """Main chat endpoint with intent routing, Graph-RAG, memory, and citation check."""
    if req.remember:
        from memory import add_fact
        add_fact(req.remember)

    user = _resolve_processing_user(vaaani_session)
    structured = wants_structured_output(req.query, STRUCTURED_TRIGGERS)
    facts_used = top_relevant_facts(req.query, retriever.embed)

    guardrail_prompt = build_universal_guardrail_prompt()
    student_guardrails: dict | None = None
    if user:
        try:
            student_guardrails = get_student_guardrails(user["id"])
            school_prompt = build_guardrail_prompt(student_guardrails)
            if school_prompt:
                guardrail_prompt = school_prompt + "\n\n" + guardrail_prompt
        except Exception:
            pass

    result, retrieval = _run_intent(
        req.query, structured,
        socratic=req.socratic,
        user=user,
        guardrail_prompt=guardrail_prompt,
        source_filter=req.source_filter,
    )
    record_query(req.query)

    guardrail_violations: list[dict] = []
    guardrail_active = bool(student_guardrails and not student_guardrails.get("allow_direct_answers", False))
    if guardrail_active:
        try:
            v = check_guardrail_violation(
                req.query, result.answer, student_guardrails, socratic_override=req.socratic,
            )
            if v and v.get("violations"):
                guardrail_violations = v["violations"]
            log_guardrail_event(
                user_id=user["id"] if user else None,
                school_id=None,
                event_type="chat_checked",
                detail=f"violations={len(guardrail_violations)} intent={result.intent}",
            )
        except Exception:
            pass

    # Post-process [[PLOT:{...}]] markers the LLM may have emitted: render each
    # to a PNG under data/figures/ and rewrite the answer to carry [[FIG:id]]
    # sentinels. Structured (JSON) answers skip this — the frontend renders
    # them as tables, not free text.
    figures: list[PlotFigure] = []
    if not structured and result.answer:
        rewritten, rendered = render_diagrams(result.answer)
        result.answer = rewritten
        figures = [
            PlotFigure(id=f.id, url=f.url, caption=f.caption, expr=f.expr)
            for f in rendered
        ]

    sources = [
        {
            "source": c.get("source", ""),
            "score": float(c.get("score", 0.0)),
            "snippet": (c.get("text", "") or "")[:240],
        }
        for c in retrieval["chunks"]
    ]
    communities = [
        {
            "id": getattr(c, "id", None),
            "title": getattr(c, "title", "") or f"community-{getattr(c, 'id', '?')}",
            "summary": getattr(c, "summary", ""),
            "findings": list(getattr(c, "findings", []) or []),
            "size": getattr(c, "size", len(getattr(c, "nodes", []) or [])),
        }
        for c in retrieval.get("communities", [])
    ]

    return ChatResponse(
        answer=result.answer,
        sources=sources,
        tokens=result.tokens_used,
        intent=result.intent,
        graph_mode=retrieval.get("graph_mode"),
        entities=retrieval.get("entities", []),
        topic_refs=retrieval.get("topic_refs", []),
        communities=communities,
        structured=result.structured,
        fidelity_warnings=result.fidelity_warnings,
        memory_used=facts_used,
        weak_spots=retrieval.get("weak_spots", []),
        user_signed_in=bool(user),
        hermes_corrections=[
            HermesCorrection(**c) for c in retrieval.get("hermes_corrections", [])
        ],
        figures=figures,
        guardrail_active=guardrail_active,
        guardrail_violations=guardrail_violations,
    )


def _sse(event: str, payload: dict) -> str:
    """Format an SSE frame. Each event has a name and a JSON-serialised body."""
    import json as _json
    return f"event: {event}\ndata: {_json.dumps(payload, ensure_ascii=False)}\n\n"


def _chat_stream_generator(req: "ChatRequest", user: dict):
    """Yield SSE frames for a chat turn.

    Sequence: open → retrieval → token* → done (or error). The opening frame
    is emitted *before* retrieval starts so Cloudflare receives bytes within
    its 100 s edge-timeout window even when retrieval + DeepSeek together
    take longer than that.
    """
    import json as _json
    import time as _time

    try:
        # Opening heartbeat — guarantees the edge sees bytes immediately.
        yield ": open\n\n"
        yield _sse("open", {"ts": _time.time()})

        if req.remember:
            from memory import add_fact as _add_fact
            _add_fact(req.remember)

        structured = wants_structured_output(req.query, STRUCTURED_TRIGGERS)
        facts_used = top_relevant_facts(req.query, retriever.embed)

        guardrail_prompt = build_universal_guardrail_prompt()
        student_guardrails: dict | None = None
        try:
            student_guardrails = get_student_guardrails(user["id"])
            school_prompt = build_guardrail_prompt(student_guardrails)
            if school_prompt:
                guardrail_prompt = school_prompt + "\n\n" + guardrail_prompt
        except Exception:
            pass

        # ---- Retrieval (same logic as _run_intent, inlined so we can yield
        #       a 'retrieval' SSE frame before kicking off the LLM stream) ----
        intent = classify(req.query)
        g_mode: str | None = None
        retrieval: dict = {"chunks": [], "entities": [], "communities": [], "edges": []}
        proposed_g_mode = graph_mode(req.query) if intent == "knowledge" else None

        query_vec = retriever.embed([req.query])[0]
        try:
            hermes_plan = hermes_corrector.plan(
                query_vec,
                user_id=user["id"],
                intent=intent,
                proposed_graph_mode=proposed_g_mode,
            )
        except Exception:
            hermes_plan = hermes_corrector.CorrectionPlan([], 0, 0.0, 0.0)
        correction_names = set(hermes_plan.names)

        if intent == "knowledge":
            g_mode = "global" if "upgrade_graph_global" in correction_names else proposed_g_mode

        effective_top_k = TOP_K * 2 if "broaden_retrieval" in correction_names else TOP_K

        sf_set: set[str] | None = set(req.source_filter) if req.source_filter else None
        if intent == "knowledge":
            retrieval = (
                retriever.global_graph_search(req.query)
                if g_mode == "global"
                else retriever.local_graph_search(req.query, k=effective_top_k, source_filter=sf_set)
            )
        elif intent == "task":
            retrieval["chunks"] = retriever.search(req.query, k=effective_top_k, source_filter=sf_set)

        chunks = retrieval["chunks"]
        memory_block = format_memory_block(facts_used)

        entity_displays = retrieval.get("entities", []) or []
        topic_refs: list[dict] = []
        weak: list[dict] = []
        if entity_displays:
            for d in entity_displays:
                key = learn_service.normalize_topic(d)
                if key:
                    topic_refs.append({"topic": key, "display": d})
            if topic_refs:
                try:
                    weak = learn_service.weak_spots(user["id"], [t["topic"] for t in topic_refs])
                except Exception:
                    weak = []

        graph_block = (
            build_graph_block(retrieval["entities"], retrieval["edges"], retrieval["communities"])
            if intent == "knowledge"
            else ""
        )
        if weak and req.socratic:
            bullets = "\n".join(
                f"- {w['display']} (student-rated mastery {w['mastery']:.1f}/5)" for w in weak
            )
            graph_block = (graph_block + "\n\n" if graph_block else "") + (
                "STUDENT WEAK SPOTS (from prior ratings) — bias your Socratic questions toward these:\n"
                + bullets
            )
        extra_system = (
            hermes_corrector.STRICT_GROUNDING_DIRECTIVE
            if "strict_grounding" in correction_names
            else ""
        )

        # Emit early metadata so the UI can paint intent + entity chips
        # while DeepSeek warms up.
        sources_payload = [
            {
                "source": c.get("source", ""),
                "score": float(c.get("score", 0.0)),
                "snippet": (c.get("text", "") or "")[:240],
            }
            for c in chunks
        ]
        yield _sse("retrieval", {
            "intent": intent,
            "graph_mode": g_mode,
            "entities": retrieval.get("entities", []),
            "topic_refs": topic_refs,
            "sources": sources_payload,
        })

        messages = build_prompt(
            req.query, chunks, memory_block, intent, structured,
            graph_mode=g_mode, graph_block=graph_block, socratic=req.socratic,
            extra_system=extra_system,
            guardrail_prompt=guardrail_prompt,
        )

        # ---- LLM streaming ----
        # call_deepseek(stream=True) returns a generator that yields content
        # deltas. We forward each delta as an SSE 'token' frame; CF sees a byte
        # every few milliseconds so the 100 s edge timeout never trips.
        accumulator: list[str] = []
        tokens_used = 0
        stream_gen = call_deepseek(messages, stream=True, json_mode=structured)
        try:
            while True:
                try:
                    delta = next(stream_gen)
                except StopIteration as stop:
                    final = stop.value or {}
                    tokens_used = final.get("tokens", 0) or 0
                    break
                if delta:
                    accumulator.append(delta)
                    yield _sse("token", {"delta": delta})
        except Exception as e:
            yield _sse("error", {"detail": f"LLM stream failed: {e}"})
            return

        answer = "".join(accumulator)
        record_query(req.query)

        # ---- Post-processing (same as /chat): structured parse, guardrail
        #       check, plot rewrite, citation fidelity, skill touches, Hermes
        #       trace. None of this can be streamed — runs after token stream
        #       ends and the result rides in the 'done' frame. ----
        structured_payload = maybe_parse_structured(answer) if structured else None

        guardrail_violations: list[dict] = []
        guardrail_active = bool(
            student_guardrails and not student_guardrails.get("allow_direct_answers", False)
        )
        if guardrail_active:
            try:
                v = check_guardrail_violation(
                    req.query, answer, student_guardrails, socratic_override=req.socratic,
                )
                if v and v.get("violations"):
                    guardrail_violations = v["violations"]
                log_guardrail_event(
                    user_id=user["id"],
                    school_id=None,
                    event_type="chat_checked",
                    detail=f"violations={len(guardrail_violations)} intent={intent}",
                )
            except Exception:
                pass

        figures_payload: list[dict] = []
        if not structured and answer:
            rewritten, rendered = render_diagrams(answer)
            answer = rewritten
            figures_payload = [
                {"id": f.id, "url": f.url, "caption": f.caption, "expr": f.expr}
                for f in rendered
            ]

        warnings = (
            citation_fidelity(answer, chunks)
            if intent == "knowledge" and g_mode == "local" and not req.socratic
            else []
        )

        if topic_refs:
            subject = learn_service.classify_subject(
                req.query, " ".join(t["display"] for t in topic_refs)
            )
            for t in topic_refs:
                try:
                    learn_service.upsert_skill(user["id"], t["topic"], t["display"], subject)
                except Exception:
                    pass

        try:
            hermes_store.log_trace(
                user_id=user["id"],
                query=req.query,
                embedding=query_vec,
                intent=intent,
                graph_mode=g_mode,
                num_chunks=len(chunks),
                fidelity_warnings=len(warnings),
                tokens=tokens_used,
                corrections_applied=hermes_plan.names,
            )
        except Exception:
            pass

        communities_payload = [
            {
                "id": getattr(c, "id", None),
                "title": getattr(c, "title", "") or f"community-{getattr(c, 'id', '?')}",
                "summary": getattr(c, "summary", ""),
                "findings": list(getattr(c, "findings", []) or []),
                "size": getattr(c, "size", len(getattr(c, "nodes", []) or [])),
            }
            for c in retrieval.get("communities", [])
        ]

        yield _sse("done", {
            "answer": answer,
            "sources": sources_payload,
            "tokens": tokens_used,
            "intent": intent,
            "graph_mode": g_mode,
            "entities": retrieval.get("entities", []),
            "topic_refs": topic_refs,
            "communities": communities_payload,
            "structured": structured_payload,
            "fidelity_warnings": warnings,
            "memory_used": facts_used,
            "weak_spots": weak,
            "user_signed_in": True,
            "hermes_corrections": [
                {"name": c.name, "reason": c.reason} for c in hermes_plan.corrections
            ],
            "figures": figures_payload,
            "guardrail_active": guardrail_active,
            "guardrail_violations": guardrail_violations,
        })
    except Exception as e:
        yield _sse("error", {"detail": str(e)[:500]})


@app.post("/chat/stream")
def chat_stream(
    req: ChatRequest,
    vaaani_session: str | None = Cookie(default=None, alias="vaaani_session"),
):
    """Streaming variant of /chat. Returns Server-Sent Events.

    The non-streaming /chat hits Cloudflare's 100 s edge timeout for long
    DeepSeek responses (524). This endpoint streams tokens as they arrive, so
    Cloudflare sees a byte every few milliseconds and the connection stays
    open indefinitely.
    """
    user = _resolve_processing_user(vaaani_session)
    return StreamingResponse(
        _chat_stream_generator(req, user),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            # X-Accel-Buffering disables nginx response buffering if anything
            # nginx-shaped sits in front of us; harmless otherwise.
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/graph")
def graph_endpoint(vaaani_session: str | None = Cookie(default=None, alias="vaaani_session")) -> dict:
    """Return the raw knowledge graph + community list (for inspection / viz).

    Auth-gated: graph nodes carry the user's uploaded entity vocabulary, so
    anonymous callers get an empty graph rather than leaking another user's
    constellation.
    """
    if not _resolve_user(vaaani_session):
        return {"nodes": [], "edges": [], "communities": []}
    kg = retriever.kg
    return {
        "nodes": [
            {"id": k, "display": d.get("display", k), "type": d.get("type", "unknown"),
             "community": retriever.community_idx.get(k)}
            for k, d in kg.g.nodes(data=True)
        ],
        "edges": [
            {"source": u, "target": v, "type": data.get("type", "related_to")}
            for u, v, data in kg.g.edges(data=True)
        ],
        "communities": [
            # 2026-05-28: summary + findings added so /graph-view can show
            # a meaningful side panel when the user clicks a community
            # tile, instead of a silent camera-fit zoom that looks broken.
            {
                "id": c.id,
                "title": c.title,
                "size": c.size,
                "summary": c.summary,
                "findings": c.findings,
            }
            for c in retriever.communities
        ],
    }


# Catch-all static mount so root-relative asset refs in site/*.html
# (e.g. /style.css, /main.js, /auth.js) resolve. Registered LAST so every
# explicit @app.get route above wins for its path; only unmatched paths
# fall through to disk under site/.
if SITE_DIR.exists():
    app.mount("/", StaticFiles(directory=str(SITE_DIR), html=False), name="site_root")
