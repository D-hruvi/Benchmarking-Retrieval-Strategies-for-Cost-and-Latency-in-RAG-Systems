"""
Hybrid retrieval RAG: BM25 keyword pre-filter, then embedding rerank only
within that narrowed candidate pool.

Concept, since BM25 is new to you: BM25 (Best Matching 25) is a scoring
function that ranks documents/chunks by keyword overlap with the query,
weighted so rare words count for more than common ones (the same idea as
TF-IDF, but with tuning that handles document length and diminishing
returns from repeated terms better). It does no embedding math at all --
just word counting -- so it's very fast and has zero embedding-model
cost. It also has zero understanding of synonyms or meaning.

The hybrid pattern: use BM25 first because it's cheap, to cut the field
from "every chunk in the corpus" down to a small candidate pool
(BM25_CANDIDATE_POOL_SIZE, set in config.py). THEN run the expensive,
semantically-aware embedding comparison, but only on that small pool
instead of the whole corpus. At this corpus's tiny scale (a few hundred
chunks) the compute saving is not dramatic -- the real point, and what
you should be able to explain in an interview, is that this pattern is
what makes semantic search tractable at scale (millions of chunks),
where comparing a query embedding against every single chunk embedding
would be far too slow to do live.

Trade-off to be upfront about: if the truly relevant chunk uses
different words than the query (no keyword overlap at all), BM25 might
not include it in the candidate pool in the first place, and the later
embedding rerank never gets a chance to find it -- naive dense search
over the whole corpus doesn't have this failure mode. This is exactly
why the benchmark's accuracy numbers matter: they're what actually shows
whether that trade-off costs you real accuracy on THIS corpus and THESE
questions, rather than you having to guess.
"""

import time
from dataclasses import dataclass

from . import config
from .groq_client import call_llm
from .indexer import LoadedIndexes
from .naive_rag import RAGResult


def answer_hybrid(
    query: str,
    indexes: LoadedIndexes,
    top_k: int = config.TOP_K,
    bm25_pool_size: int = config.BM25_CANDIDATE_POOL_SIZE,
) -> RAGResult:
    t0 = time.perf_counter()

    # Step 1: cheap keyword pre-filter. bm25_search returns (Chunk, score)
    # pairs; we need their positions in indexes.chunks to reconstruct
    # embeddings from FAISS without re-embedding anything.
    bm25_results = indexes.bm25_search(query, k=bm25_pool_size)
    chunk_id_to_index = {c.chunk_id: i for i, c in enumerate(indexes.chunks)}
    candidate_indices = [chunk_id_to_index[chunk.chunk_id] for chunk, _score in bm25_results]

    # Step 2: expensive embedding comparison, but only within the
    # narrowed candidate pool.
    query_vec = indexes.embed_query(query)
    reranked = indexes.dense_search_within(query_vec, candidate_indices, k=top_k)

    retrieval_latency = time.perf_counter() - t0

    context_chunks = [chunk.text for chunk, _score in reranked]
    llm_result = call_llm(query, context_chunks)

    total_latency = retrieval_latency + llm_result.latency_seconds

    return RAGResult(
        answer=llm_result.answer,
        retrieval_latency_s=retrieval_latency,
        llm_latency_s=llm_result.latency_seconds,
        total_latency_s=total_latency,
        input_tokens=llm_result.input_tokens,
        output_tokens=llm_result.output_tokens,
        cost_usd=llm_result.cost_usd,
        bm25_candidate_pool_size=bm25_pool_size,
    )
