"""
Single entry point every caller (benchmark script, Streamlit app) goes
through: answer_question(query, variant, indexes, cache). Routes to the
right variant's logic. Keeping this as one explicit if/elif -- not a
registry pattern or plugin abstraction -- on purpose, per your request
to keep wiring explainable in an interview without hidden indirection.
"""

from .indexer import LoadedIndexes
from .semantic_cache import SemanticCache
from .naive_rag import answer_naive, RAGResult
from .cached_rag import answer_cached
from .hybrid_rag import answer_hybrid

VALID_VARIANTS = ("naive", "cached", "hybrid")


def answer_question(
    query: str,
    variant: str,
    indexes: LoadedIndexes,
    cache: SemanticCache = None,
) -> RAGResult:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"variant must be one of {VALID_VARIANTS}, got {variant!r}")

    if variant == "naive":
        return answer_naive(query, indexes)

    if variant == "cached":
        if cache is None:
            raise ValueError("variant='cached' requires a SemanticCache instance")
        return answer_cached(query, indexes, cache)

    if variant == "hybrid":
        return answer_hybrid(query, indexes)
