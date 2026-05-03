"""Trace storage — SQLite + JSON persistence."""

import json
import os
import sqlite3
from pathlib import Path
from agent_eval.core.types import Trace


DB_PATH = Path(os.environ.get("AGENT_EVAL_DB", str(Path.home() / ".agent_eval.db")))


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    _ensure_tables(conn)
    return conn


def _ensure_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS traces (
            trace_id TEXT PRIMARY KEY,
            run_id TEXT,
            agent_version TEXT,
            scenario_id TEXT,
            timestamp TEXT,
            input TEXT,
            final_output TEXT,
            total_steps INTEGER,
            trace_json TEXT
        );

        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_version TEXT,
            timestamp TEXT,
            overall_score REAL,
            pre_tool_score REAL,
            at_tool_score REAL,
            end_to_end_score REAL,
            metric_scores_json TEXT
        );

        CREATE TABLE IF NOT EXISTS eval_results (
            eval_id TEXT PRIMARY KEY,
            trace_id TEXT,
            layer TEXT,
            metric TEXT,
            score REAL,
            explanation TEXT,
            details_json TEXT,
            FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
        );
    """)
    conn.commit()


def save_trace(trace: Trace):
    """Save a trace to the SQLite store."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO traces
           (trace_id, run_id, agent_version, scenario_id, timestamp,
            input, final_output, total_steps, trace_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            trace.trace_id, trace.run_id, trace.agent_version,
            trace.scenario_id, trace.timestamp,
            trace.input, trace.final_output, trace.total_steps,
            json.dumps(trace.to_dict()),
        ),
    )
    conn.commit()
    conn.close()


def load_trace(trace_id: str) -> Trace | None:
    """Load a trace from the store."""
    conn = _get_conn()
    row = conn.execute("SELECT trace_json FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
    conn.close()
    if row:
        return Trace.from_dict(json.loads(row["trace_json"]))
    return None


def load_traces_by_version(agent_version: str) -> list[Trace]:
    """Load all traces for a given agent version."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT trace_json FROM traces WHERE agent_version = ? ORDER BY timestamp",
        (agent_version,),
    ).fetchall()
    conn.close()
    return [Trace.from_dict(json.loads(r["trace_json"])) for r in rows]


def save_eval_results(results: list):
    """Save eval results to the store."""
    conn = _get_conn()
    for r in results:
        conn.execute(
            """INSERT OR REPLACE INTO eval_results
               (eval_id, trace_id, layer, metric, score, explanation, details_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                r.eval_id, r.trace_id, r.layer, r.metric,
                r.score, r.explanation, json.dumps(r.details),
            ),
        )
    conn.commit()
    conn.close()


def save_score_history(
    agent_version: str,
    overall: float,
    pre_tool: float,
    at_tool: float,
    end_to_end: float,
    metric_scores: dict,
):
    """Save aggregate scores for an agent version."""
    from datetime import datetime
    conn = _get_conn()
    conn.execute(
        """INSERT INTO scores
           (agent_version, timestamp, overall_score, pre_tool_score,
            at_tool_score, end_to_end_score, metric_scores_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            agent_version, datetime.now().isoformat(),
            overall, pre_tool, at_tool, end_to_end,
            json.dumps(metric_scores),
        ),
    )
    conn.commit()
    conn.close()


def get_score_history() -> list[dict]:
    """Get all score history records."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM scores ORDER BY timestamp").fetchall()
    conn.close()
    return [dict(r) for r in rows]
