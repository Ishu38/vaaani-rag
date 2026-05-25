"""Hermes pre-flight corrector.

For each incoming /chat query, look up k nearest past traces. If those
neighbours show a failure pattern, return concrete corrections the dispatcher
can apply *before* the LLM call:

  - upgrade_graph_global  →  force graph_mode='global' for knowledge intent
                              (broader community context when local-RAG kept
                              hitting fidelity warnings)
  - broaden_retrieval     →  retrieve extra chunks (top_k bumped)
  - strict_grounding      →  append a directive to the system prompt telling
                              the model to refuse claims not in the chunks

Corrections are *advisory* — the dispatcher decides whether each applies
given the current intent. We never silently rewrite the user's query.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hermes.store import Trace, nearest_traces


# Tunables — small constants kept here, not in config.py, because they are
# specific to the Hermes policy and likely to evolve as we collect traces.
NEIGHBOUR_K = 8
NEIGHBOUR_MIN_SIM = 0.55
FAIL_RATE_THRESHOLD = 0.4        # ≥40% neighbours flagged → upgrade retrieval
SPARSE_CHUNK_THRESHOLD = 2.0     # avg < 2 chunks across neighbours → broaden
WARNING_NEIGHBOUR_COUNT = 2      # ≥2 neighbours with warnings → strict mode


@dataclass
class Correction:
    """A single advisory adjustment, with a human-readable reason."""
    name: str       # machine label (matches Hermes vocab)
    reason: str     # one-line "why" for /hermes/recent and the chat UI


@dataclass
class CorrectionPlan:
    """All corrections Hermes suggests for one /chat call, plus its evidence."""
    corrections: list[Correction]
    neighbours_considered: int
    neighbour_fail_rate: float
    neighbour_avg_chunks: float

    @property
    def names(self) -> list[str]:
        """Just the machine labels — what gets persisted in the trace log."""
        return [c.name for c in self.corrections]


def plan(
    embedding: np.ndarray,
    *,
    user_id: int | None,
    intent: str,
    proposed_graph_mode: str | None,
) -> CorrectionPlan:
    """Inspect past traces and return a CorrectionPlan for this query.

    `proposed_graph_mode` is what intent.graph_mode() suggested. Hermes may
    upgrade local→global if neighbours kept failing; it never downgrades.
    """
    neighbours: list[Trace] = nearest_traces(
        embedding, user_id=user_id, k=NEIGHBOUR_K, min_similarity=NEIGHBOUR_MIN_SIM,
    )
    if not neighbours:
        return CorrectionPlan([], 0, 0.0, 0.0)

    failed = [n for n in neighbours if n.fidelity_warnings > 0]
    fail_rate = len(failed) / len(neighbours)
    avg_chunks = sum(n.num_chunks for n in neighbours) / len(neighbours)

    out: list[Correction] = []

    # Rule 1 — upgrade local→global for knowledge queries with a bad neighbour history.
    # (Only applies to knowledge — task/meta/calendar don't use the graph at all.)
    if (
        intent == "knowledge"
        and proposed_graph_mode == "local"
        and fail_rate >= FAIL_RATE_THRESHOLD
    ):
        out.append(Correction(
            name="upgrade_graph_global",
            reason=(
                f"{len(failed)}/{len(neighbours)} similar past queries hit "
                "fidelity warnings under local retrieval — escalating to "
                "global community context."
            ),
        ))

    # Rule 2 — broaden retrieval when neighbours typically returned sparse chunks.
    # Applies anywhere the retriever actually runs (knowledge + task).
    if intent in ("knowledge", "task") and avg_chunks < SPARSE_CHUNK_THRESHOLD:
        out.append(Correction(
            name="broaden_retrieval",
            reason=(
                f"Past similar queries averaged {avg_chunks:.1f} chunks "
                "(< 2) — widening top_k to reduce blank-answer risk."
            ),
        ))

    # Rule 3 — strict grounding when failures keep recurring in this neighbourhood.
    # Applies to any intent whose chunks feed the LLM (knowledge + task);
    # meta/calendar don't ground on chunks, so the directive would be a no-op there.
    if intent in ("knowledge", "task") and len(failed) >= WARNING_NEIGHBOUR_COUNT:
        out.append(Correction(
            name="strict_grounding",
            reason=(
                f"{len(failed)} prior neighbours produced unsupported "
                "sentences — appending a strict-grounding directive to the prompt."
            ),
        ))

    return CorrectionPlan(
        corrections=out,
        neighbours_considered=len(neighbours),
        neighbour_fail_rate=round(fail_rate, 3),
        neighbour_avg_chunks=round(avg_chunks, 2),
    )


STRICT_GROUNDING_DIRECTIVE = (
    "STRICT GROUNDING: every factual claim in your answer must be traceable "
    "to the chunks above. If the chunks do not contain the answer, say "
    "exactly: \"I don't have enough in your library to answer that.\" Do not "
    "use outside knowledge."
)
