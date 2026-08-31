"""
Runs every question in questions.json through naive, cached, and hybrid
RAG, logging one row per (question, variant) to results.db, then
exports results.csv.

Per your working style: run this for ONE variant at a time and grade
that variant's rows before moving to the next, rather than running all
three blind. Use --variant to control that:

    python scripts/run_benchmark.py --variant naive
    # ... go grade results.csv by hand ...
    python scripts/run_benchmark.py --variant cached
    # ... go grade again ...
    python scripts/run_benchmark.py --variant hybrid

Cached RAG's results depend on run ORDER (later questions can hit the
cache from earlier ones), so re-running --variant cached alone after
naive/hybrid have already run will still behave correctly -- it only
reads/writes semantic_cache.db, which is separate from results.db.

Run with no --variant to run all three back to back (only recommended
once you've already graded each variant individually at least once).
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexer import LoadedIndexes
from src.semantic_cache import SemanticCache
from src.pipeline import answer_question, VALID_VARIANTS
from src.logging_db import get_connection, log_result, export_csv
from src import config


def load_questions():
    with open(config.QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run(variants):
    print("Loading indexes (embedding model + FAISS + BM25)...")
    indexes = LoadedIndexes()
    cache = SemanticCache() if "cached" in variants else None

    questions = load_questions()
    conn = get_connection()

    total = len(questions) * len(variants)
    done = 0

    for variant in variants:
        print(f"\n=== Running variant: {variant} ({len(questions)} questions) ===")
        for q in questions:
            result = answer_question(q["question"], variant, indexes, cache=cache)
            done += 1
            print(f"[{done}/{total}] {q['id']} ({variant}) "
                  f"-- {result.total_latency_s:.2f}s, ${result.cost_usd:.6f}"
                  + (f", cache_hit={result.cache_hit}" if result.cache_hit is not None else ""))

            log_result(conn, {
                "question_id": q["id"],
                "question": q["question"],
                "question_type": q["type"],
                "variant": variant,
                "answer": result.answer,
                "retrieval_latency_s": result.retrieval_latency_s,
                "llm_latency_s": result.llm_latency_s,
                "total_latency_s": result.total_latency_s,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": result.cost_usd,
                "cache_hit": result.cache_hit,
                "cache_similarity_to_nearest": result.cache_similarity_to_nearest,
                "bm25_candidate_pool_size": result.bm25_candidate_pool_size,
            })

    conn.close()
    csv_path = export_csv()
    print(f"\nDone. {done} rows logged. Open {csv_path} and fill in the "
          f"'correctness' column by hand (correct / partial / wrong) before "
          f"moving to analysis.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=VALID_VARIANTS, default=None,
        help="Run only this variant. Omit to run all three.",
    )
    args = parser.parse_args()
    variants = [args.variant] if args.variant else list(VALID_VARIANTS)
    run(variants)
