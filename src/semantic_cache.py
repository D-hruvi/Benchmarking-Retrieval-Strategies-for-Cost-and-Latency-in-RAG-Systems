"""
Semantic cache for the cached-RAG variant.

Concept: instead of caching by exact query string match (which only helps
if someone asks the *identical* question twice), we cache by MEANING.
Every incoming query is embedded, and compared against the embeddings of
all previously-cached queries using cosine similarity. If the closest
past query is similar enough (above CACHE_SIMILARITY_THRESHOLD), we
reuse its stored answer instead of running retrieval + an LLM call again.
That's the entire mechanism -- no separate "cache key hashing" needed
beyond the embedding itself, since the embedding IS the fuzzy key.

Storage: a local SQLite file (cache_store.db), per your no-Redis
constraint. Also kept in memory as numpy arrays during a session for fast
similarity search without hitting disk on every query -- the DB is the
persistence layer, the in-memory arrays are the query-time working set.
"""

import sqlite3
import json

import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS semantic_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    embedding TEXT NOT NULL,   -- JSON-encoded list of floats
    answer TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


class SemanticCache:
    def __init__(self, db_path: str = config.CACHE_DB_PATH, threshold: float = config.CACHE_SIMILARITY_THRESHOLD):
        self.db_path = db_path
        self.threshold = threshold
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(SCHEMA)
        self.conn.commit()
        self._load_into_memory()

    def _load_into_memory(self):
        rows = self.conn.execute("SELECT id, query, embedding, answer FROM semantic_cache").fetchall()
        self._ids = [r[0] for r in rows]
        self._queries = [r[1] for r in rows]
        self._answers = [r[3] for r in rows]
        if rows:
            self._embeddings = np.array([json.loads(r[2]) for r in rows], dtype="float32")
        else:
            self._embeddings = np.zeros((0, config.EMBEDDING_DIM), dtype="float32")

    def lookup(self, query_vec: np.ndarray):
        """
        query_vec: normalized embedding, shape (1, dim).
        Returns (hit: bool, answer_or_None, best_similarity: float,
        matched_query_or_None). best_similarity is returned even on a
        miss (or when the cache is empty, in which case it's -1.0) so
        callers can log it for later threshold tuning, per your spec.
        """
        if self._embeddings.shape[0] == 0:
            return False, None, -1.0, None

        # embeddings are normalized (see indexer.py), so a plain dot
        # product against normalized query_vec gives cosine similarity
        # directly -- no need to divide by norms again.
        sims = self._embeddings @ query_vec[0]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.threshold:
            return True, self._answers[best_idx], best_sim, self._queries[best_idx]
        return False, None, best_sim, self._queries[best_idx]

    def store(self, query: str, query_vec: np.ndarray, answer: str):
        embedding_json = json.dumps(query_vec[0].tolist())
        self.conn.execute(
            "INSERT INTO semantic_cache (query, embedding, answer) VALUES (?, ?, ?)",
            (query, embedding_json, answer),
        )
        self.conn.commit()
        self._load_into_memory()  # keep in-memory arrays in sync; cheap
        # at this corpus/query-volume scale -- would need a smarter
        # incremental append if this were caching millions of queries.

    def size(self) -> int:
        return len(self._ids)
