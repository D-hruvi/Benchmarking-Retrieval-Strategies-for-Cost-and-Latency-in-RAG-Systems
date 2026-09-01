"""
Index building: turns the chunked corpus into two search structures that
get built ONCE (by scripts/build_index.py) and then just loaded at
runtime by every variant.

Why two indexes:
  - FAISS (dense/embedding index): every chunk is embedded into a 384-dim
    vector by the sentence-transformers model. A query is embedded the
    same way, and FAISS finds the chunks whose vectors are closest to the
    query vector (cosine similarity, via normalized vectors + inner
    product search). This captures *semantic* similarity -- it can match
    a query about "automobile" to a chunk about "car" even with zero
    shared words.
  - BM25 (keyword index): a classic term-frequency ranking algorithm
    (the theoretical upgrade of TF-IDF used in real search engines). No
    embeddings involved -- pure word matching, weighted by how rare/common
    each word is across the corpus. It's cheap and fast, but blind to
    synonyms. The hybrid variant uses this to cheaply narrow the field
    before doing the expensive embedding comparison on a smaller set.

Both indexes are built from the exact same chunk list, so a chunk_index
position means the same thing in both.
"""

import os
import json
import pickle

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from . import config
from .loader import Chunk, load_and_chunk_all


def get_embedding_model() -> SentenceTransformer:
    """
    Loads the sentence-transformers model. This is the single most
    memory-heavy object in the whole system (~90MB for MiniLM-L6-v2 plus
    PyTorch's own overhead) -- see scripts/build_index.py's memory report
    for the measured footprint.
    """
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


def _tokenize_for_bm25(text: str) -> list:
    """Simple whitespace + lowercase tokenizer. BM25 doesn't need anything
    fancier than this for a corpus this size -- stemming/lemmatization
    would help marginally but adds a dependency and complexity that isn't
    worth it here."""
    return text.lower().split()


def build_indexes(data_dir: str = config.DATA_DIR, index_dir: str = config.INDEX_DIR):
    """
    Build the FAISS dense index and the BM25 keyword index from data_dir,
    and persist both (plus chunk metadata) to index_dir so they can be
    committed to the repo and loaded at container startup without
    rebuilding -- this is the Render free-tier constraint from the brief:
    no assumption of a persistent disk between deploys, so the index
    must either be pre-built and committed, or rebuilt fast on startup.
    Rebuilding from 20 small .txt files takes seconds, so either approach
    works; this project commits the pre-built index for faster cold starts.
    """
    os.makedirs(index_dir, exist_ok=True)

    print("Loading and chunking documents...")
    chunks = load_and_chunk_all(data_dir)
    print(f"  {len(chunks)} chunks from {len(set(c.doc_id for c in chunks))} documents")

    print("Loading embedding model (this downloads ~90MB on first run)...")
    model = get_embedding_model()

    print("Embedding all chunks...")
    texts = [c.text for c in chunks]
    embeddings = model.encode(
        texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    )
    # normalize_embeddings=True makes inner product == cosine similarity,
    # so we can use FAISS's fast IndexFlatIP instead of computing cosine
    # similarity manually.
    embeddings = embeddings.astype("float32")

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, config.FAISS_INDEX_PATH)

    print("Building BM25 index...")
    tokenized_corpus = [_tokenize_for_bm25(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)

    print("Saving chunk metadata...")
    chunks_meta = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "text": c.text,
            "chunk_index": c.chunk_index,
            "token_count": c.token_count,
        }
        for c in chunks
    ]
    with open(config.CHUNKS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks_meta, f, indent=2)

    print(f"Done. Index artifacts written to {index_dir}/")
    return chunks, faiss_index, bm25


class LoadedIndexes:
    """Holds everything a variant needs at query time, loaded once and
    reused across all queries -- this is the object every variant's
    retrieval function is handed."""

    def __init__(self):
        if not (
            os.path.exists(config.CHUNKS_METADATA_PATH)
            and os.path.exists(config.FAISS_INDEX_PATH)
            and os.path.exists(config.BM25_INDEX_PATH)
        ):
            # Self-healing fallback: index_store/ is supposed to be built
            # once locally and committed to the repo (see README), but if
            # it's missing at runtime -- e.g. it was never committed, or
            # a hosting platform's build didn't preserve it -- build it
            # right here instead of crashing. Render's free tier has no
            # persistent disk between deploys but DOES have full internet
            # at container startup, and rebuilding from 20 short .txt
            # files takes seconds, so this is a cheap, safe fallback
            # rather than a silent failure mode that depends on a human
            # not forgetting a manual step before every deploy.
            print(
                f"index_store/ not found or incomplete at {config.INDEX_DIR} "
                f"-- building it now from data/ (this happens once per "
                f"container; takes a few seconds for this corpus size)."
            )
            build_indexes()

        with open(config.CHUNKS_METADATA_PATH, "r", encoding="utf-8") as f:
            chunks_meta = json.load(f)
        self.chunks = [Chunk(**c) for c in chunks_meta]

        self.faiss_index = faiss.read_index(config.FAISS_INDEX_PATH)

        with open(config.BM25_INDEX_PATH, "rb") as f:
            self.bm25 = pickle.load(f)

        self.embedding_model = get_embedding_model()

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.embedding_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        return vec.astype("float32")

    def dense_search(self, query_vec: np.ndarray, k: int) -> list:
        """Returns list of (Chunk, score) for the top-k nearest chunks in
        the full FAISS index."""
        scores, indices = self.faiss_index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def bm25_search(self, query: str, k: int) -> list:
        """Returns list of (Chunk, bm25_score) for the top-k chunks by
        BM25 keyword score."""
        tokenized_query = _tokenize_for_bm25(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]

    def dense_search_within(self, query_vec: np.ndarray, candidate_indices: list, k: int) -> list:
        """
        Re-rank a small candidate pool (e.g. from BM25) by embedding
        similarity, WITHOUT re-searching the full FAISS index. This is
        the "cheap keyword pre-filter narrows the field, then expensive
        embedding comparison only runs on the narrowed field" step that
        makes hybrid retrieval cheaper than naive dense search over the
        whole corpus -- for a corpus this small the saving is marginal,
        but the pattern is what matters and is what scales.
        """
        candidate_vecs = np.stack(
            [self._get_chunk_vector(i) for i in candidate_indices]
        )
        sims = candidate_vecs @ query_vec[0]
        order = np.argsort(sims)[::-1][:k]
        return [(self.chunks[candidate_indices[i]], float(sims[i])) for i in order]

    def _get_chunk_vector(self, chunk_idx: int) -> np.ndarray:
        """Reconstructs a single chunk's embedding vector from the FAISS
        index (IndexFlatIP stores vectors verbatim, so this is a cheap
        lookup, not a re-embedding call)."""
        return self.faiss_index.reconstruct(int(chunk_idx))


if __name__ == "__main__":
    build_indexes()
