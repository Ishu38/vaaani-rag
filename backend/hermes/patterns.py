"""Aggregate analytics over the hermes_traces log.

Cheap SQL — no in-memory clustering — because the traces table stays small
(personal RAG, single-user). If this ever needs to scale, swap in a real
embedding cluster (HDBSCAN over the BLOB column).
"""
from __future__ import annotations

from auth.db import connect
from hermes.store import init_hermes_db


def overall_stats(user_id: int | None = None) -> dict:
    """Top-of-dashboard numbers: total traces, fidelity rate, avg chunks, intent mix."""
    init_hermes_db()
    where = "WHERE user_id = ?" if user_id is not None else ""
    params: tuple = (user_id,) if user_id is not None else ()
    with connect() as c:
        total = c.execute(
            f"SELECT COUNT(*) AS n FROM hermes_traces {where}", params
        ).fetchone()["n"]
        if total == 0:
            return {
                "total_traces": 0,
                "fidelity_failure_rate": 0.0,
                "avg_chunks": 0.0,
                "corrections_applied_rate": 0.0,
                "intent_breakdown": {},
            }
        failed = c.execute(
            f"SELECT COUNT(*) AS n FROM hermes_traces {where} "
            f"{'AND' if where else 'WHERE'} fidelity_warnings > 0",
            params,
        ).fetchone()["n"]
        corrected = c.execute(
            f"SELECT COUNT(*) AS n FROM hermes_traces {where} "
            f"{'AND' if where else 'WHERE'} corrections_applied != '[]'",
            params,
        ).fetchone()["n"]
        avg_chunks = c.execute(
            f"SELECT AVG(num_chunks) AS m FROM hermes_traces {where}", params
        ).fetchone()["m"] or 0.0
        intent_rows = c.execute(
            f"SELECT intent, COUNT(*) AS n FROM hermes_traces {where} GROUP BY intent",
            params,
        ).fetchall()
    return {
        "total_traces": int(total),
        "fidelity_failure_rate": round(failed / total, 3),
        "avg_chunks": round(float(avg_chunks), 2),
        "corrections_applied_rate": round(corrected / total, 3),
        "intent_breakdown": {r["intent"]: int(r["n"]) for r in intent_rows},
    }


def weak_query_templates(user_id: int | None = None, limit: int = 10) -> list[dict]:
    """Queries that have repeatedly produced fidelity warnings — the ones
    Hermes most wants to fix. Deduped by lower-cased query string."""
    init_hermes_db()
    where = "WHERE user_id = ?" if user_id is not None else ""
    params: tuple = (user_id,) if user_id is not None else ()
    with connect() as c:
        rows = c.execute(
            f"""SELECT LOWER(query) AS q,
                       COUNT(*) AS attempts,
                       SUM(CASE WHEN fidelity_warnings > 0 THEN 1 ELSE 0 END) AS warnings,
                       MAX(created_at) AS last_seen
                FROM hermes_traces
                {where}
                GROUP BY LOWER(query)
                HAVING warnings > 0
                ORDER BY warnings DESC, attempts DESC
                LIMIT ?""",
            (*params, limit),
        ).fetchall()
    return [
        {
            "query": r["q"],
            "attempts": int(r["attempts"]),
            "warnings": int(r["warnings"]),
            "warning_rate": round(int(r["warnings"]) / int(r["attempts"]), 3),
            "last_seen": r["last_seen"],
        }
        for r in rows
    ]


def correction_effectiveness(user_id: int | None = None) -> list[dict]:
    """For each correction name, how often it fired and the fidelity-failure rate
    of traces where it fired — lets us see whether the policy is actually helping."""
    init_hermes_db()
    where = "WHERE user_id = ?" if user_id is not None else ""
    params: tuple = (user_id,) if user_id is not None else ()
    with connect() as c:
        rows = c.execute(
            f"""SELECT corrections_applied AS j,
                       fidelity_warnings > 0 AS failed
                FROM hermes_traces
                {where}""",
            params,
        ).fetchall()
    import json
    bucket: dict[str, dict] = {}
    for r in rows:
        names = json.loads(r["j"] or "[]")
        for n in names:
            b = bucket.setdefault(n, {"fired": 0, "failed": 0})
            b["fired"] += 1
            if r["failed"]:
                b["failed"] += 1
    return [
        {
            "name": name,
            "fired": v["fired"],
            "post_fail_rate": round(v["failed"] / v["fired"], 3) if v["fired"] else 0.0,
        }
        for name, v in sorted(bucket.items(), key=lambda kv: -kv[1]["fired"])
    ]
