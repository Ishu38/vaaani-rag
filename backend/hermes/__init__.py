"""Hermes — self-correcting layer for the RAG pipeline.

Logs every /chat trace (query embedding + retrieval shape + fidelity warnings)
and, before each new query, looks up its k-nearest past traces. If neighbours
failed (fidelity warnings, sparse retrieval), it returns concrete corrections
that the dispatcher applies before calling the LLM.

Modules:
  store      — SQLite trace log + k-NN over embeddings (lives in users.db)
  corrector  — pre-flight policy: trace neighbours → list[Correction]
  patterns   — aggregate analytics for /hermes/* dashboards
  routes     — FastAPI router (/hermes/stats, /hermes/recent, /hermes/patterns)
"""
from hermes.store import init_hermes_db, log_trace, nearest_traces

__all__ = ["init_hermes_db", "log_trace", "nearest_traces"]
