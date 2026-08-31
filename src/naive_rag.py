"""
Naive RAG: the baseline. Embed the query, retrieve top-k chunks by dense
(embedding) similarity search over the WHOLE corpus, send them to the
LLM, return the answer. No caching, no keyword pre-filtering -- every
query pays full retrieval + full LLM cost. This is what "cached" and
"hybrid" are each optimizing away one piece of.
"""

import time
from dataclasses import dataclass

from . import config
from .groq_client import call_llm
from .indexer import LoadedIndexes


@dataclass
class RAGResult:
    answer: str
    retrieval_latency_s: float
    llm_latency_s: float
    total_latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cache_hit: int = None
    cache_similarity_to_nearest: float = None
    bm25_candidate_pool_size: int = None


def answer_naive(query: str, indexes: LoadedIndexes, top_k: int = config.TOP_K) -> RAGResult:
    t0 = time.perf_counter()
    query_vec = indexes.embed_query(query)
    retrieved = indexes.dense_search(query_vec, k=top_k)
    retrieval_latency = time.perf_counter() - t0

    context_chunks = [chunk.text for chunk, _score in retrieved]
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
    )
