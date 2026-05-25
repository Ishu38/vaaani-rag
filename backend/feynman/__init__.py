"""Feynman-diff: explain-it-back tester for ingested documents.

Student writes a free-form explanation of a topic. We extract the entities
and relations they mentioned (via the same DeepSeek pass we use for graph
construction), then diff the result against the k-hop subgraph around the
chosen topic node. The output names exactly which nodes/edges the student
covered, which they missed, and where their relations don't match the
graph's. This is the moat: nobody else can do this without an entity graph
over the user's own corpus.
"""
from .differ import (
    FeynmanResult,
    diff_explanation,
    list_topics,
    summarize_diff,
)

__all__ = [
    "FeynmanResult",
    "diff_explanation",
    "list_topics",
    "summarize_diff",
]
