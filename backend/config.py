"""Central configuration. All paths resolve relative to the project root."""
from __future__ import annotations
import os
from pathlib import Path

# Tell HuggingFace transformers (loaded transitively by sentence-transformers)
# not to import TensorFlow or Flax. The miniconda tensorflow 2.21 install on
# this host has a broken .so symbol; PyTorch alone is enough for embeddings.
# Must be set BEFORE `transformers` is imported anywhere.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_PATH = DATA_DIR / "index.tq"
METADATA_PATH = DATA_DIR / "metadata.json"
MEMORY_PATH = DATA_DIR / "memory.json"

# Embedding model
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
BIT_WIDTH = 4

# Chunking
CHUNK_TOKENS = 512
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 5
MEMORY_TOP_K = 3
MAX_RECENT_QUERIES = 20
# Relevance floor for knowledge answers. Vector search always returns the
# nearest chunks even for questions the corpus can't answer; below this
# cosine score a chunk is noise and the honest no-context reply is better
# than letting the small model improvise. Measured on MiniLM-L6-v2:
# on-topic queries hit ≥0.41, off-topic ≤0.17 — 0.30 splits with margin.
MIN_RELEVANCE = float(os.environ.get("VAAANI_MIN_RELEVANCE", "0.30"))

# Vaaani's own local LLM engine (OpenAI-compatible) — REPLACES DeepSeek.
# Served by vaaani-model/serve_openai.py (Qwen2.5-3B-Instruct 4-bit + the grade-spiral LoRA).
#   VAAANI_LLM_MODEL      -> "vaaani-base"     : base Qwen for general RAG / chat / ingest
#   VAAANI_FLAGSHIP_MODEL -> "vaaani-flagship" : base + curriculum adapter (Root-Bridge tutor)
VAAANI_LLM_BASE_URL = os.environ.get("VAAANI_LLM_BASE_URL", "http://127.0.0.1:8011")
VAAANI_LLM_API_KEY = os.environ.get("VAAANI_LLM_API_KEY", "local")  # any non-empty; engine ignores it
VAAANI_LLM_MODEL = os.environ.get("VAAANI_LLM_MODEL", "vaaani-base")
VAAANI_FLAGSHIP_MODEL = os.environ.get("VAAANI_FLAGSHIP_MODEL", "vaaani-flagship")
# 300 not 180: the Qwen3-4B base generates ~30% slower per token on CPU than
# the old 3B; long grounded answers need the headroom (streaming path unaffected).
VAAANI_LLM_TIMEOUT = int(os.environ.get("VAAANI_LLM_TIMEOUT", "300"))

# Backward-compat aliases: the old DEEPSEEK_* names now point at the local Vaaani
# engine, so every existing call site (chat, ingest, community, classifier) is
# repointed with no code change and NO DeepSeek endpoint is ever contacted.
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", VAAANI_LLM_BASE_URL)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", VAAANI_LLM_API_KEY)
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", VAAANI_LLM_MODEL)
DEEPSEEK_INGEST_MODEL = os.environ.get("DEEPSEEK_INGEST_MODEL", "") or VAAANI_LLM_MODEL
DEEPSEEK_TIMEOUT = VAAANI_LLM_TIMEOUT

# Intelligence layer
STRUCTURED_TRIGGERS = ("give me a table", "compare", "list with details")
INTENTS = ("knowledge", "task", "calendar", "meta")

# Graph-RAG (Microsoft pattern)
GRAPH_PATH = DATA_DIR / "graph.json"
COMMUNITIES_PATH = DATA_DIR / "communities.json"
GRAPH_CACHE_PATH = DATA_DIR / "graph_cache.json"
# Cap per-community size when asking DeepSeek to summarise, to keep prompts sane.
COMMUNITY_MAX_NODES = 60
# When answering "global" queries, how many top community summaries to use.
GLOBAL_TOP_COMMUNITIES = 5
# Local graph expansion: 1-hop neighbors of entities found in retrieved chunks.
LOCAL_HOPS = 1

# ----- Auth -----
USERS_DB_PATH = DATA_DIR / "users.db"
JWT_SECRET = os.environ.get("JWT_SECRET", "")  # MUST be set in production
JWT_ALGO = "HS256"
JWT_EXP_DAYS = 30
COOKIE_NAME = "vaaani_session"
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "0") == "1"  # set to 1 behind HTTPS
# Share the login cookie across subdomains (app.vaaani.in ↔ api.vaaani.in) so the
# Vercel frontend and the GCP backend see the same session. Empty = host-only
# (local dev). In prod set COOKIE_DOMAIN=.vaaani.in, COOKIE_SECURE=1.
COOKIE_DOMAIN = os.environ.get("COOKIE_DOMAIN", "").strip()
COOKIE_SAMESITE = os.environ.get("COOKIE_SAMESITE", "lax").strip().lower()
# Cross-origin browsers require explicit origins (not "*") when cookies are sent.
# Comma-separated, e.g. "https://app.vaaani.in,https://vaaani.in". Empty = permissive
# local dev (allow all, no credentials).
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
EMAIL_VERIFY_EXP_HOURS = 48
PHONE_OTP_EXP_MIN = 10
PHONE_OTP_LEN = 6

# Google OAuth (optional — button hidden if either missing)
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "http://127.0.0.1:8765/auth/google/callback")

# GitHub OAuth (optional — button hidden if either missing)
GITHUB_CLIENT_ID = os.environ.get("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.environ.get("GITHUB_CLIENT_SECRET", "")
GITHUB_REDIRECT_URI = os.environ.get("GITHUB_REDIRECT_URI", "http://127.0.0.1:8765/auth/github/callback")

# SMTP (optional — falls back to console logging)
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
SMTP_FROM = os.environ.get("SMTP_FROM", "no-reply@vaaani.in")
SMTP_USE_TLS = os.environ.get("SMTP_USE_TLS", "1") == "1"
APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://127.0.0.1:8765")

# SMS / OTP (optional — falls back to console logging)
SMS_PROVIDER = os.environ.get("SMS_PROVIDER", "")  # 'msg91' | 'twilio' | '' (console)
MSG91_AUTH_KEY = os.environ.get("MSG91_AUTH_KEY", "")
MSG91_SENDER_ID = os.environ.get("MSG91_SENDER_ID", "VAAANI")
MSG91_TEMPLATE_ID = os.environ.get("MSG91_TEMPLATE_ID", "")
TWILIO_SID = os.environ.get("TWILIO_SID", "")
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN", "")
TWILIO_FROM = os.environ.get("TWILIO_FROM", "")

# Pricing plan flag
DEFAULT_PLAN = "free"

# ----- Audio narration (Piper TTS, CPU-only) -----
AUDIO_DIR = DATA_DIR / "audio"
TTS_VOICES_DIR = DATA_DIR / "tts_voices"
# Default voice. Override with VAAANI_TTS_VOICE env var (filename without
# extension, must exist as .onnx + .onnx.json in TTS_VOICES_DIR).
TTS_DEFAULT_VOICE = os.environ.get("VAAANI_TTS_VOICE", "en_GB-jenny_dioco-medium")
# Path to the piper CLI. Falls back to PATH lookup if unset.
PIPER_BIN = os.environ.get("PIPER_BIN", "piper")
# Per-paragraph cap (chars). Long inputs are split on blank lines to keep
# piper's per-call working set bounded on the shared VPS.
TTS_PARAGRAPH_CHARS = 1500
# MP3 bitrate for narration output. Mono speech sounds clean at 64-96 kbps.
TTS_MP3_BITRATE = "80k"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TTS_VOICES_DIR.mkdir(parents=True, exist_ok=True)

# ----- Ingest limits -----
# Maximum upload size for /ingest endpoints. Cloudflare free tier = 100 MB;
# we cap at 50 MB to keep chunk counts bounded and leave headroom.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

# File extensions that go through OCR (image → text) rather than direct text
# extraction. Tesseract handles PNG, JPEG; WEBP gets Pillow-converted to JPEG
# first. These are included in SUPPORTED_EXT by ingest.py.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

for p in (RAW_DIR, DATA_DIR):
    p.mkdir(parents=True, exist_ok=True)
