"""
Cached RAG: naive RAG + a semantic cache in front of it.

Flow: embed the query once. Check the semantic cache for a prior query
above the similarity threshold. On a hit, skip retrieval AND the LLM
call entirely -- return the cached answer, near-zero latency and cost.
On a miss, fall through to full naive RAG, then store the new
query/answer pair in the cache for future hits.

Note the query embedding is computed exactly once and reused for both
the cache lookup and (on a miss) the retrieval step -- no reason to
embed the same string twice.
"""

import time
from dataclasses import replace

from . import config
from .groq_client import call_llm
from .indexer import LoadedIndexes
from .naive_rag import RAGResult
from .semantic_cache import SemanticCache


def answer_cached(
    query: str,
    indexes: LoadedIndexes,
    cache: SemanticCache,
    top_k: int = config.TOP_K,
) -> RAGResult:
    t0 = time.perf_counter()
    query_vec = indexes.embed_query(query)
    embed_latency = time.perf_counter() - t0

    hit, cached_answer, similarity, _matched_query = cache.lookup(query_vec)

    if hit:
        # Cache hit: no retrieval beyond the query embedding itself, no
        # LLM call at all -- this is the "near-zero cost/latency" path
        # the brief asks for. We still report the embedding time honestly
        # rather than claiming literal zero latency, since embedding the
        # incoming query is unavoidable work even on a hit.
        return RAGResult(
            answer=cached_answer,
            retrieval_latency_s=embed_latency,
            llm_latency_s=0.0,
            total_latency_s=embed_latency,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            cache_hit=1,
            cache_similarity_to_nearest=similarity,
        )

    # Cache miss: run retrieval (reusing the already-computed query_vec,
    # not re-embedding) + LLM call, same as naive RAG.
    t1 = time.perf_counter()
    retrieved = indexes.dense_search(query_vec, k=top_k)
    retrieval_latency = embed_latency + (time.perf_counter() - t1)

    context_chunks = [chunk.text for chunk, _score in retrieved]
    llm_result = call_llm(query, context_chunks)

    cache.store(query, query_vec, llm_result.answer)

    total_latency = retrieval_latency + llm_result.latency_seconds

    return RAGResult(
        answer=llm_result.answer,
        retrieval_latency_s=retrieval_latency,
        llm_latency_s=llm_result.latency_seconds,
        total_latency_s=total_latency,
        input_tokens=llm_result.input_tokens,
        output_tokens=llm_result.output_tokens,
        cost_usd=llm_result.cost_usd,
        cache_hit=0,
        cache_similarity_to_nearest=similarity,
    )
