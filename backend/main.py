"""FastAPI app exposing /ingest, /status and the static frontend."""
from __future__ import annotations

import shutil
import sqlite3 as _sqlite3
from pathlib import Path

from fastapi import Cookie, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
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
from cognitive.fingerprint import build_fingerprint
from cognitive_loop_routes import router as cognitive_loop_router
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
    MIN_RELEVANCE,
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
    scrub_provider_identity,
)
from memory import (
    format_memory_block,
    load_memory,
    record_query,
    top_relevant_facts,
)
from retriever import Retriever
import developmental_firewall

app = FastAPI(title="Local RAG Assistant", version="0.1.0")
# In prod, CORS_ORIGINS names the exact frontend origins (e.g. https://app.vaaani.in)
# and credentials are allowed so the shared cookie flows. Locally (no CORS_ORIGINS)
# we stay permissive without credentials — the browser rule forbids "*" + cookies.
from config import CORS_ORIGINS as _CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS or ["*"],
    allow_credentials=bool(_CORS_ORIGINS),
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(learning_router)
app.include_router(hermes_router)
app.include_router(messenger_router)
app.include_router(youtube_router)
app.include_router(cognitive_router)
app.include_router(cognitive_loop_router)
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

# Phase 3: Preload the graph cache at startup for O(1) deterministic
# word lookups.  If the cache doesn't exist (first run), rebuild it
# from the knowledge graph.
try:
    import graph_cache
    cache = graph_cache.load_cache()
    if not cache:
        graph_cache.rebuild_and_save()
except Exception:
    pass

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



@app.get("/")
def root() -> FileResponse:
    """Serve the marketing landing page. Falls back to chat if the site isn't built."""
    if SITE_INDEX.exists():
        return FileResponse(SITE_INDEX)
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    raise HTTPException(404, "Neither site/index.html nor frontend/index.html found")



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


@app.get("/roots")
def roots_page() -> FileResponse:
    """Serve the Word Roots module (morphology-first: roots/affixes before sound)."""
    return _serve_site("roots.html")


# ─────────────────── Active learning on the graph ───────────────────────────

class FixitCheckBody(BaseModel):
    id: str = Field(..., min_length=1, max_length=40)
    idx: int = Field(..., ge=0, le=40)


class BuildCheckBody(BaseModel):
    sentence: str = Field(default="", max_length=400)
    targets: list[str] = Field(default_factory=list)


@app.get("/learning/fixit")
def learning_fixit(
    exclude: str | None = None,
    vaaani_session: str | None = Cookie(default=None, alias="vaaani_session"),
) -> dict:
    """A 'spot the slip' challenge — error hidden; the child detects it."""
    _resolve_processing_user(vaaani_session)  # sign-in required
    import active_learning
    return active_learning.fixit_next(exclude)


@app.post("/learning/fixit/check")
def learning_fixit_check(
    body: FixitCheckBody,
    vaaani_session: str | None = Cookie(default=None, alias="vaaani_session"),
) -> dict:
    user = _resolve_processing_user(vaaani_session)
    import active_learning
    return active_learning.fixit_check(body.id, body.idx, student_id=str(user["id"]))


@app.post("/learning/build/check")
def learning_build_check(
    body: BuildCheckBody,
    vaaani_session: str | None = Cookie(default=None, alias="vaaani_session"),
) -> dict:
    """The child built a sentence from their own graph words — reward it."""
    user = _resolve_processing_user(vaaani_session)
    import active_learning
    return active_learning.build_check(body.sentence, body.targets, student_id=str(user["id"]))


@app.get("/feel")
def feel_page() -> FileResponse:
    """Feel the Sound — render a phonological feature (voicing) as a haptic
    buzz on the device's vibration actuator: the 'Feel' stage of the Language
    Journey, made physical on existing mobile hardware."""
    return _serve_site("feel.html")


@app.get("/missions")
def missions_page() -> FileResponse:
    """CASCADE smoke test — deterministic discovery missions, no LLM."""
    return _serve_site("missions.html")


@app.get("/ipa")
def ipa_page() -> FileResponse:
    """Serve the interactive IPA chart (phonetics learning tool)."""
    return _serve_site("ipa.html")


@app.get("/language-map")
def language_map_page() -> FileResponse:
    """Serve the grade-gated Language Map (question-first developmental graph + assessment pyramid)."""
    return _serve_site("language-map.html")


@app.get("/evolution")
def evolution_page() -> FileResponse:
    """Serve the Phoneme Evolution game (mutate sound-creatures, fill in the IPA)."""
    return _serve_site("evolution.html")


@app.get("/sound-lab")
def sound_lab_page() -> FileResponse:
    """Sound Lab — live Web Audio formant synthesis: drag the vowel space to hear
    a vowel morph, toggle voicing, and hear+feel real speech sounds on-device."""
    return _serve_site("sound-lab.html")


@app.get("/game")
def game_alias() -> RedirectResponse:
    """Friendly alias → the phoneme game."""
    return RedirectResponse(url="/evolution", status_code=302)


@app.get("/sw.js")
def service_worker() -> FileResponse:
    """Serve the service worker with no-cache so SW updates reach clients
    immediately — never let the edge or browser pin an old shell/cache."""
    p = SITE_DIR / "sw.js"
    if not p.exists():
        raise HTTPException(404, "sw.js not found")
    return FileResponse(
        p,
        media_type="text/javascript",
        headers={"Cache-Control": "no-cache, max-age=0, must-revalidate"},
    )


@app.get("/pricing")
def pricing_page() -> RedirectResponse:
    """Pricing has been removed from the product surface — send visitors home."""
    return RedirectResponse(url="/", status_code=302)


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
    status_user = _resolve_user(vaaani_session)
    if not status_user:
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
    # Privacy scope: counts, document list, graph figures, memory and recent
    # queries are all restricted to what THIS user may see.
    s = retriever.status(allowed_paths=_allowed_paths(status_user))
    from memory import _facts_for, recent_queries_for
    mem = load_memory()
    s["memory_facts"] = len(_facts_for(mem, status_user["id"]))
    s["recent_queries"] = recent_queries_for(mem, status_user["id"], n=5)
    return s


class NarrateRequest(BaseModel):
    doc_name: str
    voice: str | None = None
    # "narration" → single-voice readback; "podcast" → 2-host dialogue
    mode: str = "narration"


def _allowed_doc_names(user: dict | None) -> set[str] | None:
    """Display names of the documents `user` may read (None = unrestricted)."""
    allowed = _allowed_paths(user)
    if allowed is None:
        return None
    files = retriever.metadata.get("files", {})
    return {v.get("name", "") for k, v in files.items() if k in allowed}


@app.get("/audio/library")
def audio_library(vaaani_session: str | None = Cookie(default=None, alias="vaaani_session")) -> dict:
    """Ingested docs eligible for narration plus the voice list.

    Auth-gated + privacy-scoped: only the caller's visible documents appear.
    """
    lib_user = _resolve_user(vaaani_session)
    if not lib_user:
        return {"docs": [], "voices": available_voices()}
    names = _allowed_doc_names(lib_user)
    docs = list_narratable_docs()
    if names is not None:
        docs = [d for d in docs if d.get("doc_name") in names]
    return {
        "docs": docs,
        "voices": available_voices(),
    }


@app.post("/audio/narrate")
def audio_narrate(
    req: NarrateRequest,
    vaaani_session: str | None = Cookie(default=None, alias="vaaani_session"),
) -> dict:
    """Synthesize an MP3 for an ingested document. Idempotent via SHA1 cache."""
    narrate_user = _resolve_user(vaaani_session)
    if not narrate_user:
        raise HTTPException(401, "Sign in to use Vaaani.")
    names = _allowed_doc_names(narrate_user)
    if names is not None and req.doc_name not in names:
        # 404, not 403 — don't confirm the document exists for other users.
        raise HTTPException(404, f"no ingested document named '{req.doc_name}'")
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
def feynman_topics(vaaani_session: str | None = Cookie(default=None, alias="vaaani_session")) -> dict:
    """Topics worth explaining back: well-connected graph nodes, ranked
    by degree desc. Auth-gated + scoped to the caller's visible documents."""
    fey_user = _resolve_user(vaaani_session)
    if not fey_user:
        return {"topics": []}
    allowed = _allowed_paths(fey_user)
    topics = [t for t in list_topics() if retriever.node_visible(t.get("id", ""), allowed)]
    return {"topics": topics}


@app.post("/feynman/diff")
def feynman_diff(
    req: FeynmanRequest,
    vaaani_session: str | None = Cookie(default=None, alias="vaaani_session"),
) -> dict:
    """Run the explain-it-back diff against the corpus subgraph for the
    chosen topic. Returns structured node/edge coverage."""
    fey_user = _resolve_user(vaaani_session)
    if not fey_user:
        raise HTTPException(401, "Sign in to use Vaaani.")
    if not retriever.node_visible(req.topic_id, _allowed_paths(fey_user)):
        # 404, not 403 — don't confirm the topic exists for other users.
        raise HTTPException(404, f"unknown topic '{req.topic_id}'")
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
    ingest_user = _resolve_processing_user(vaaani_session)
    name = Path(file.filename or "upload.bin").name
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {sorted(SUPPORTED_EXT)}")
    # Per-user subdirectory: same-named files from different users get
    # distinct file keys, and ownership is unambiguous.
    user_dir = RAW_DIR / f"u{ingest_user['id']}"
    user_dir.mkdir(parents=True, exist_ok=True)
    dest = user_dir / name
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    size_bytes = dest.stat().st_size
    if size_bytes > MAX_UPLOAD_BYTES:
        dest.unlink(missing_ok=True)
        raise HTTPException(413, f"File too large: {size_bytes/1e6:.1f} MB. Max: {MAX_UPLOAD_BYTES/1e6:.0f} MB")
    summary = ingest(RAW_DIR, INDEX_PATH, METADATA_PATH)
    import scope
    scope.record_ownership(
        str(dest.resolve()), ingest_user["id"], scope.sharing_school_ids(ingest_user)
    )
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
    async_user = _resolve_processing_user(vaaani_session)
    name = Path(file.filename or "upload.bin").name
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {sorted(SUPPORTED_EXT)}")
    user_dir = RAW_DIR / f"u{async_user['id']}"
    user_dir.mkdir(parents=True, exist_ok=True)
    dest = user_dir / name
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    size_bytes = dest.stat().st_size
    if size_bytes > MAX_UPLOAD_BYTES:
        dest.unlink(missing_ok=True)
        raise HTTPException(413, f"File too large: {size_bytes/1e6:.1f} MB. Max: {MAX_UPLOAD_BYTES/1e6:.0f} MB")
    # Ownership is recorded up-front (idempotent) so the file is scoped even
    # if the worker crashes mid-ingest and retries later.
    import scope
    scope.record_ownership(
        str(dest.resolve()), async_user["id"], scope.sharing_school_ids(async_user)
    )

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


def _discovery_retrieval_query(ctx: dict) -> str:
    """Synthesize a KB retrieval query from the learner's discovery state.

    The orchestrator fast-path doesn't have a user query — the learner just
    arrived. But the KB must still reason about what the learner should
    discover next. This function turns the discovery_context (mastered sounds,
    weak patterns, word families, current unit) into a query string that
    local_graph_search can use to retrieve relevant chunks, entities, and
    communities from the knowledge graph.

    Priority: weak_patterns (highest leverage) → mastered sounds + families
    → current unit. Keeps the query short (Graph-RAG works best with focused
    queries, not kitchen-sink ones).
    """
    parts: list[str] = []

    # Weak patterns are the ZPD frontier — highest retrieval priority
    for w in (ctx.get("weak_patterns") or ctx.get("current_weak_areas") or [])[:2]:
        # "ph -> /f/" becomes "ph f" — search for both the grapheme and phoneme
        cleaned = w.replace("→", " ").replace("->", " ").replace("/", " ")
        cleaned = " ".join(cleaned.split())  # collapse whitespace
        if cleaned:
            parts.append(cleaned)

    # Recent errors give more signal about what to retrieve
    for e in (ctx.get("recent_errors") or ctx.get("recently_confused_concepts") or [])[:2]:
        cleaned = e.replace("/", " ").replace("→", " ")
        cleaned = " ".join(cleaned.split())
        if cleaned:
            parts.append(cleaned)

    # Mastered sounds + their word families anchor the familiar 80%
    mastered = ctx.get("mastered_sounds") or []
    families = ctx.get("unlocked_word_families") or []
    if families:
        parts.append(f"word family {' '.join(families[:4])}")
    elif mastered:
        parts.append(f"words with {' '.join(mastered[:4])} sounds")

    # Current unit / stage
    unit = ctx.get("current_unit") or ctx.get("current_stage") or ""
    if unit:
        parts.append(unit)

    return " ".join(parts) if parts else "English sounds spelling patterns phonics"


def _try_graph_route(query: str, *, grade: int | None = None,
                    user: dict | None = None):
    """Try to answer a query from the structural linguistics graph.

    Returns a graph_router.GraphResult (confidence 0–5), or None if routing
    errors.  Caller should only use answers with confidence >= 4 for a
    deterministic response; confidence 2-3 can be appended as context to the
    LLM prompt; confidence 0-1 falls through to normal retrieval.
    """
    from graph_router import route_query
    g = grade or (int(user.get("grade", 2)) if user else 2)
    return route_query(query, g)


def _allowed_paths(user: dict | None) -> set[str] | None:
    """The set of document file-keys `user` may read (None = unrestricted,
    only when scoping is disabled). See scope.py for the sharing rules."""
    import scope
    return scope.allowed_paths_for(user, retriever.metadata.get("files", {}))


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

    Used by gated endpoints (/ingest, /ingest/async).
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


def build_learner_profile_block(user: dict | None) -> str:
    """Compact 'what Vaaani already knows about this student' block, injected
    into every answer so the tutor adapts without the learner re-explaining
    themselves each session (Uday/SWI: 'pre-trained, knows the student').

    Built from the persistent cognitive fingerprint. Returns '' for guests and
    for brand-new students with no history yet, so the tutor stays neutral until
    there is something real to adapt to.
    """
    if not user:
        return ""
    try:
        fp = build_fingerprint(user["id"])
    except Exception:
        return ""
    s = fp.get("summary", {}) or {}
    if not s.get("total_analyzed"):
        return ""
    lines = [
        "STUDENT PROFILE — you already know this learner from past sessions. "
        "Adapt difficulty, examples and tone to them; do NOT make them re-explain "
        "what they know:",
    ]
    name = user.get("name") or user.get("display_name")
    if name:
        lines.append(f"- Name: {name}.")
    lines.append(f"- Has answered {s['total_analyzed']} questions, ~{s.get('accuracy', 0)}% correct.")
    if fp.get("strengths"):
        lines.append(f"- Already strong in: {', '.join(fp['strengths'][:4])} (acknowledge, don't over-drill).")
    if fp.get("weaknesses"):
        lines.append(f"- Needs practice in: {', '.join(fp['weaknesses'][:4])} (gently steer practice here).")
    pw = s.get("primary_weakness_label")
    if pw and pw not in ("None", "No data yet", "Unknown"):
        lines.append(f"- Most common mistake type: {pw} — address the root cause, not just the symptom.")
    biases = fp.get("biases", {}) or {}
    if biases.get("description"):
        lines.append(f"- Tendency: {biases['description']}.")
    if biases.get("speed_issue"):
        lines.append(f"- {biases['speed_issue']} — nudge them to check their work before answering.")
    res = fp.get("resilience_score")
    if isinstance(res, (int, float)) and res < 0.4:
        lines.append("- Confidence is fragile right now — be encouraging and scaffold in small steps.")
    lines.append(
        "Use this only to calibrate how you teach. Never read this profile back to the student."
    )
    return "\n".join(lines)


def _run_intent(
    query: str,
    structured: bool,
    *,
    socratic: bool = False,
    user: dict | None = None,
    guardrail_prompt: str = "",
    source_filter: list[str] | None = None,
    allowed_paths: set[str] | None = None,
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
            retriever.global_graph_search(query, allowed_paths=allowed_paths)
            if g_mode == "global"
            else retriever.local_graph_search(
                query, k=effective_top_k, source_filter=sf_set, allowed_paths=allowed_paths
            )
        )
    elif intent == "task":
        retrieval["chunks"] = retriever.search(
            query, k=effective_top_k, source_filter=sf_set, allowed_paths=allowed_paths
        )

    # Relevance floor (local mode): nearest-neighbour search returns chunks
    # for ANY query; below MIN_RELEVANCE they are noise, and graph context
    # extracted from noise chunks is noise too — clear it so the no-context
    # gate below can fire instead of the model improvising.
    if intent == "knowledge" and g_mode != "global":
        kept = [c for c in retrieval["chunks"] if float(c.get("score", 0.0)) >= MIN_RELEVANCE]
        if kept:
            retrieval = {**retrieval, "chunks": kept}
        else:
            retrieval = {**retrieval, "chunks": [], "entities": [], "edges": [], "communities": []}

    chunks = retrieval["chunks"]
    facts = top_relevant_facts(query, retriever.embed, user_id=user["id"] if user else None)
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

    # Hard gate: a knowledge question with NOTHING retrieved (no chunks, no
    # graph context) is the worst hallucination case for the small model —
    # don't generate at all, answer honestly and deterministically. The trace
    # is still logged so Hermes learns these queries are chunk-starved.
    if intent == "knowledge" and not chunks and not (
        retrieval.get("entities") or retrieval.get("edges") or retrieval.get("communities")
    ):
        try:
            hermes_store.log_trace(
                user_id=user_id,
                query=query,
                embedding=query_vec,
                intent=intent,
                graph_mode=g_mode,
                num_chunks=0,
                fidelity_warnings=0,
                tokens=0,
                corrections_applied=hermes_plan.names,
            )
        except Exception:
            pass
        return LLMResponse(
            answer=(
                "I don't have anything about that in my knowledge base yet. "
                "Try uploading the relevant notes or textbook chapter, or ask me "
                "about something from the material that's already been added."
            ),
            sources_used=[],
            tokens_used=0,
            intent=intent,
        ), {
            **retrieval,
            "graph_mode": g_mode,
            "topic_refs": topic_refs,
            "weak_spots": weak,
            "hermes_corrections": [
                {"name": c.name, "reason": c.reason} for c in hermes_plan.corrections
            ],
        }

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
    # Persistent learner memory: the tutor already knows this student.
    profile_block = build_learner_profile_block(user)
    if profile_block:
        extra_system = (extra_system + "\n\n" + profile_block) if extra_system else profile_block
    messages = build_prompt(
        query, chunks, memory_block, intent, structured,
        graph_mode=g_mode, graph_block=graph_block, socratic=socratic,
        extra_system=extra_system,
        guardrail_prompt=guardrail_prompt,
    )
    # Greedy decoding for factual answers; keep light sampling for Socratic
    # questioning and creative/task intents.
    temperature = 0.0 if intent == "knowledge" and not socratic else 0.2
    resp = call_deepseek(messages, stream=False, json_mode=structured, temperature=temperature)

    choice = resp.get("choices", [{}])[0]
    answer = scrub_provider_identity(choice.get("message", {}).get("content", ""))
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


@app.get("/graph")
def graph_endpoint(vaaani_session: str | None = Cookie(default=None, alias="vaaani_session")) -> dict:
    """Return the raw knowledge graph + community list (for inspection / viz).

    Auth-gated: graph nodes carry the user's uploaded entity vocabulary, so
    anonymous callers get an empty graph rather than leaking another user's
    constellation.
    """
    graph_user = _resolve_user(vaaani_session)
    if not graph_user:
        return {"nodes": [], "edges": [], "communities": [], "journey": {}}
    kg = retriever.kg
    # Privacy scope: only nodes extracted from files this user may read, edges
    # between two visible nodes, and communities with at least one visible node.
    allowed = _allowed_paths(graph_user)
    visible = {
        k for k in kg.g.nodes if retriever.node_visible(k, allowed)
    }

    # LIVING-BRAIN OVERLAY: a star is not just an entity from a document — it is
    # a concept THIS learner has acquired. Join each node to the learner's
    # skill row (student_skills.topic == the normalized graph-node id) so the
    # node carries mastery (brightness), whether it's fading (due for review),
    # and when it was discovered. Data already computed by the practice/review
    # engine; this is the wiring that makes the graph a cognitive map.
    skills_by_topic: dict[str, dict] = {}
    due_topics: set[str] = set()
    try:
        for s in learn_service.list_skills(graph_user["id"], limit=1000):
            skills_by_topic[s["topic"]] = s
        for d in learn_service.due_for_review(graph_user["id"], limit=1000):
            due_topics.add(d["topic"])
    except Exception:
        pass

    nodes = []
    for k, d in kg.g.nodes(data=True):
        if k not in visible:
            continue
        skill = skills_by_topic.get(k)
        nodes.append({
            "id": k,
            "display": d.get("display", k),
            "type": d.get("type", "unknown"),
            "community": retriever.community_idx.get(k),
            # None when the concept is in the corpus but not yet practised by
            # this learner (an unlit star — there to be discovered).
            "discovered": skill is not None,
            "mastery": round(skill["mastery"], 2) if skill else None,
            "interval_days": skill.get("interval_days") if skill else None,
            "due": k in due_topics,
            "last_seen": skill.get("last_seen_at") if skill else None,
        })

    discovered = [n for n in nodes if n["discovered"]]
    avg_mastery = (sum(n["mastery"] for n in discovered) / len(discovered)) if discovered else 0.0
    recent = sorted(discovered, key=lambda n: n["last_seen"] or "", reverse=True)[:8]
    journey = {
        "concepts": len(nodes),
        "discovered": len(discovered),
        "connections": sum(1 for u, v in ((e[0], e[1]) for e in kg.g.edges) if u in visible and v in visible),
        "constellations": sum(1 for c in retriever.communities if retriever.community_visible(c, allowed)),
        "discovery_pct": round(100 * len(discovered) / len(nodes), 1) if nodes else 0.0,
        "memory_health": round(avg_mastery / 5.0, 3),   # 0..1
        "due_count": sum(1 for n in discovered if n["due"]),
        "recently_discovered": [{"display": n["display"], "mastery": n["mastery"]} for n in recent],
    }

    communities_out = [
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
        if retriever.community_visible(c, allowed)
    ]
    return {
        "nodes": nodes,
        "edges": [
            {"source": u, "target": v, "type": data.get("type", "related_to")}
            for u, v, data in kg.g.edges(data=True)
            if u in visible and v in visible
        ],
        "communities": communities_out,
        "journey": journey,
    }


@app.get("/graph/cache")
def graph_cache_endpoint(
    vaaani_session: str | None = Cookie(default=None, alias="vaaani_session"),
) -> dict:
    """Return the precomputed graph cache — O(1) word breakdowns for every
    word and root in the knowledge graph. Loaded at startup; rebuilt when
    graph_seeder.py runs. Does NOT require auth (the cache is curriculum
    data, not user data).
    """
    from graph_cache import load_cache
    cache = load_cache()
    return {
        "stats": cache.get("stats", {}),
        "roots": cache.get("roots", {}),
        "phonemes": cache.get("phonemes", {}),
        "graphemes": cache.get("graphemes", {}),
        "indexes": cache.get("indexes", {}),
        # Don't expose full word cache via API — it's large and used
        # internally by the graph router. But expose a summary.
        "word_count": len(cache.get("words", {})),
        "word_list": sorted(cache.get("words", {}).keys()),
    }


# Catch-all static mount so root-relative asset refs in site/*.html
# (e.g. /style.css, /main.js, /auth.js) resolve. Registered LAST so every
# explicit @app.get route above wins for its path; only unmatched paths
# fall through to disk under site/.
if SITE_DIR.exists():
    app.mount("/", StaticFiles(directory=str(SITE_DIR), html=False), name="site_root")
