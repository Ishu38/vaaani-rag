"""SQLite persistence for users + verification tokens + OTPs.

Plain stdlib sqlite3 — no ORM, no migration framework. Schema is created on
first call to `init_db()`; future migrations should be additive (ALTER TABLE).
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from config import USERS_DB_PATH

_LOCK = threading.Lock()
_INITIALIZED = False


def _connect(path: Path = USERS_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with row-as-dict access and FK enforcement."""
    conn = sqlite3.connect(str(path), detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


@contextmanager
def connect(path: Path = USERS_DB_PATH):
    """Context-managed connection; commits on clean exit, rolls back on error."""
    conn = _connect(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT,                   -- NULL for OAuth-only accounts
    name TEXT,
    phone TEXT,
    plan TEXT NOT NULL DEFAULT 'free',
    google_sub TEXT UNIQUE,               -- Google subject id for linked accounts
    github_id TEXT UNIQUE,                -- GitHub numeric user id for linked accounts
    email_verified_at TEXT,               -- ISO-8601; NULL means unverified
    phone_verified_at TEXT,               -- ISO-8601; NULL means unverified
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_login_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_users_email ON users(email);

CREATE TABLE IF NOT EXISTS email_verification_tokens (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL,
    used_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_evt_user ON email_verification_tokens(user_id);

CREATE TABLE IF NOT EXISTS phone_otps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    phone TEXT NOT NULL,
    code_hash TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    used_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_otp_user ON phone_otps(user_id, used_at);

-- Multi-tenant school org model. A school licenses student accounts under one
-- org. Memberships link existing users to a school with a role.
CREATE TABLE IF NOT EXISTS schools (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    code TEXT NOT NULL UNIQUE,             -- invite code used to join (e.g. "SWI-DEL-001")
    plan TEXT NOT NULL DEFAULT 'school_trial',
    guardrails TEXT NOT NULL DEFAULT '{}',  -- JSON: curriculum, socratic_level, etc.
    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_schools_code ON schools(code);

CREATE TABLE IF NOT EXISTS school_memberships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    school_id INTEGER NOT NULL REFERENCES schools(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('admin','teacher','student','parent')),
    joined_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, school_id)
);
CREATE INDEX IF NOT EXISTS ix_memb_user ON school_memberships(user_id);
CREATE INDEX IF NOT EXISTS ix_memb_school ON school_memberships(school_id);

CREATE TABLE IF NOT EXISTS parent_student_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    school_id INTEGER NOT NULL REFERENCES schools(id) ON DELETE CASCADE,
    linked_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(parent_user_id, student_user_id, school_id)
);
CREATE INDEX IF NOT EXISTS ix_psl_parent ON parent_student_links(parent_user_id, school_id);
CREATE INDEX IF NOT EXISTS ix_psl_student ON parent_student_links(student_user_id, school_id);

-- Adaptive learning: one row per (user, topic). 'topic' is the canonical
-- entity key from the Graph-RAG knowledge graph (e.g. "saturn", "hyperion").
CREATE TABLE IF NOT EXISTS student_skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    topic TEXT NOT NULL,
    display TEXT NOT NULL,                -- human-friendly name (entity display)
    subject TEXT,                          -- 'physics' | 'math' | 'english' | 'writing' | ...
    mastery REAL NOT NULL DEFAULT 2.0,      -- clamped to [0, 5]
    interval_days REAL NOT NULL DEFAULT 1.0, -- SM-2-lite review spacing
    attempts INTEGER NOT NULL DEFAULT 0,
    last_seen_at TEXT NOT NULL DEFAULT (datetime('now')),
    due_at TEXT NOT NULL DEFAULT (datetime('now', '+1 day')),
    UNIQUE(user_id, topic)
);
CREATE INDEX IF NOT EXISTS ix_skill_due ON student_skills(user_id, due_at);
CREATE INDEX IF NOT EXISTS ix_skill_mastery ON student_skills(user_id, mastery);

-- Every rating the student gives. Keeps a per-attempt audit so we can
-- visualise progress over time.
CREATE TABLE IF NOT EXISTS student_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    topic TEXT NOT NULL,
    rating INTEGER NOT NULL,              -- -1 confused | 0 sort-of | +1 got-it
    query TEXT,
    at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_att_user ON student_attempts(user_id, at);
"""


def _additive_migrate(conn: sqlite3.Connection) -> None:
    """ALTER TABLE for columns added after a database was first created.

    SQLite ignores `IF NOT EXISTS` on ALTER TABLE ADD COLUMN, so we introspect
    the schema and only add missing columns. Keep all migrations here so the
    db file stays forward-compatible across upgrades.
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "github_id" not in cols:
        # CREATE UNIQUE INDEX serves as the UNIQUE constraint we'd have written
        # inline in CREATE TABLE for a fresh database.
        conn.execute("ALTER TABLE users ADD COLUMN github_id TEXT")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_users_github_id ON users(github_id) WHERE github_id IS NOT NULL")

    # 2025 — school org tier: add schools + memberships tables if they don't exist yet.
    tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}
    if "schools" not in tables:
        conn.execute("""
            CREATE TABLE schools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                code TEXT NOT NULL UNIQUE,
                plan TEXT NOT NULL DEFAULT 'school_trial',
                guardrails TEXT NOT NULL DEFAULT '{}',
                created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS ix_schools_code ON schools(code)")
    if "school_memberships" not in tables:
        conn.execute("""
            CREATE TABLE school_memberships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                school_id INTEGER NOT NULL REFERENCES schools(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK(role IN ('admin','teacher','student','parent')),
                joined_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(user_id, school_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS ix_memb_user ON school_memberships(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_memb_school ON school_memberships(school_id)")
    if "parent_student_links" not in tables:
        conn.execute("""
            CREATE TABLE parent_student_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                student_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                school_id INTEGER NOT NULL REFERENCES schools(id) ON DELETE CASCADE,
                linked_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(parent_user_id, student_user_id, school_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS ix_psl_parent ON parent_student_links(parent_user_id, school_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_psl_student ON parent_student_links(student_user_id, school_id)")

    # 2026-05-26 — DPDP Act 2023 scaffolding. Adds DOB + consent_status to
    # users, plus parental_consents (one row per child / version) and an
    # immutable dpdp_audit_log. Verifiable parent identity (Aadhaar/DigiLocker)
    # is NOT wired here — see [[feedback_vaaani_cf_streaming]] sibling notes.
    if "date_of_birth" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN date_of_birth TEXT")  # ISO YYYY-MM-DD
    if "consent_status" not in cols:
        # not_required = 18+ at signup; pending = under-18 awaiting parent;
        # granted = parent confirmed; withdrawn = parent revoked.
        conn.execute("ALTER TABLE users ADD COLUMN consent_status TEXT NOT NULL DEFAULT 'not_required'")
    if "active_consent_id" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN active_consent_id INTEGER")
    if "deleted_at" not in cols:
        # Soft-delete marker for §12 erasure. Hard rows scrubbed by a separate job;
        # marking deleted_at suspends all auth + DPDP-gated access immediately.
        conn.execute("ALTER TABLE users ADD COLUMN deleted_at TEXT")

    if "parental_consents" not in tables:
        conn.execute("""
            CREATE TABLE parental_consents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                child_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                parent_name TEXT,
                parent_email TEXT NOT NULL,
                parent_phone TEXT,
                consent_token TEXT NOT NULL UNIQUE,
                consent_text_version TEXT NOT NULL,
                requested_at TEXT NOT NULL DEFAULT (datetime('now')),
                granted_at TEXT,
                withdrawn_at TEXT,
                parent_ip TEXT,
                parent_user_agent TEXT,
                expires_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS ix_pc_token ON parental_consents(consent_token)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_pc_child ON parental_consents(child_user_id)")

    if "dpdp_audit_log" not in tables:
        conn.execute("""
            CREATE TABLE dpdp_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                event_type TEXT NOT NULL,
                detail TEXT,
                actor_ip TEXT,
                actor_user_agent TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS ix_dpdp_audit_user ON dpdp_audit_log(user_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_dpdp_audit_event ON dpdp_audit_log(event_type, created_at)")


def init_db(path: Path = USERS_DB_PATH) -> None:
    """Create tables on first call + run any additive migrations. Idempotent."""
    global _INITIALIZED
    with _LOCK:
        if _INITIALIZED:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with connect(path) as conn:
            conn.executescript(SCHEMA)
            _additive_migrate(conn)
        _INITIALIZED = True
