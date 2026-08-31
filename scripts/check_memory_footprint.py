"""
Reports the process's resident memory (RSS) with the embedding model,
FAISS index, and BM25 index ALL loaded at once -- i.e. the exact memory
state the Streamlit app will be in on Render right after startup. Run
this BEFORE building out the naive/cached/hybrid logic and the Streamlit
app, per the brief -- catching a memory ceiling problem here is cheap;
catching it after a failed Render deploy is not.

Usage: python scripts/check_memory_footprint.py

Requires index_store/ to already exist (run build_index.py first).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil

process = psutil.Process(os.getpid())


def rss_mb() -> float:
    return process.memory_info().rss / (1024 * 1024)


if __name__ == "__main__":
    baseline = rss_mb()
    print(f"Baseline RSS (Python interpreter, before any imports): {baseline:.1f} MB")

    from src.indexer import LoadedIndexes

    after_import = rss_mb()
    print(f"RSS after importing torch/sentence-transformers/faiss: {after_import:.1f} MB "
          f"(+{after_import - baseline:.1f} MB)")

    indexes = LoadedIndexes()

    after_load = rss_mb()
    print(f"RSS after loading embedding model + FAISS index + BM25 index: "
          f"{after_load:.1f} MB (+{after_load - after_import:.1f} MB for this step, "
          f"+{after_load - baseline:.1f} MB total)")

    print()
    print(f"Chunks loaded: {len(indexes.chunks)}")
    print(f"FAISS index size: {indexes.faiss_index.ntotal} vectors x "
          f"{indexes.faiss_index.d} dims")
    print()
    print("--- Interpretation ---")
    print(f"Total RSS: {after_load:.1f} MB.")
    print("Render's free tier caps instance memory at 512MB. If the number "
          "above is uncomfortably close to that (e.g. above ~350-400MB), "
          "leave headroom before adding the Streamlit app on top -- Streamlit "
          "itself, plus request handling, adds its own overhead beyond this "
          "baseline. If it's tight, the two levers are: swap to an even "
          "smaller embedding model, or reduce corpus/chunk count (both cut "
          "into what this benchmark measures, so treat them as a last resort).")
