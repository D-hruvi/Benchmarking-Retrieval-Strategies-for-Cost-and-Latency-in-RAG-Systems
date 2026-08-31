"""
Run this once to build the FAISS + BM25 indexes from data/ and commit the
resulting index_store/ directory to your repo.

Usage: python scripts/build_index.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexer import build_indexes

if __name__ == "__main__":
    build_indexes()
