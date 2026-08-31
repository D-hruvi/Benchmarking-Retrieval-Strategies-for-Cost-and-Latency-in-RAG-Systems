"""
Results logging: one row per (question, variant) pair, per your spec.

SQLite is used as the source of truth (easy to query, no external DB
needed, bundles fine in the Render container). results.csv is exported
from it as the deliverable format for RESULTS.md / spreadsheet grading.
"""

import sqlite3
import csv

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id TEXT NOT NULL,
    question TEXT NOT NULL,
    question_type TEXT NOT NULL,
    variant TEXT NOT NULL,               -- naive / cached / hybrid
    answer TEXT,
    retrieval_latency_s REAL,
    llm_latency_s REAL,
    total_latency_s REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    cache_hit INTEGER,                   -- 0/1, NULL for non-cached variants
    cache_similarity_to_nearest REAL,    -- logged even on miss, per spec
    bm25_candidate_pool_size INTEGER,    -- NULL for non-hybrid variants
    correctness TEXT,                    -- filled in manually: correct / partial / wrong
    run_timestamp TEXT DEFAULT (datetime('now'))
);
"""


def get_connection(db_path: str = config.RESULTS_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def log_result(conn: sqlite3.Connection, row: dict):
    """row keys must match the columns above (correctness/run_timestamp
    are optional -- correctness defaults to NULL and is filled in later
    by hand, run_timestamp auto-fills)."""
    columns = [
        "question_id", "question", "question_type", "variant", "answer",
        "retrieval_latency_s", "llm_latency_s", "total_latency_s",
        "input_tokens", "output_tokens", "cost_usd", "cache_hit",
        "cache_similarity_to_nearest", "bm25_candidate_pool_size",
    ]
    values = [row.get(c) for c in columns]
    placeholders = ", ".join(["?"] * len(columns))
    conn.execute(
        f"INSERT INTO results ({', '.join(columns)}) VALUES ({placeholders})",
        values,
    )
    conn.commit()


def export_csv(db_path: str = config.RESULTS_DB_PATH, csv_path: str = config.RESULTS_CSV_PATH):
    conn = get_connection(db_path)
    cursor = conn.execute("SELECT * FROM results ORDER BY id")
    col_names = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    conn.close()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(col_names)
        writer.writerows(rows)
    print(f"Exported {len(rows)} rows to {csv_path}")
    return csv_path


if __name__ == "__main__":
    # Quick sanity check: creates the DB/table if not present, reports row count.
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    print(f"results.db has {count} rows.")
