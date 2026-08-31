"""
Streamlit demo: type a question, pick a variant (or run all 3 side by
side), see the answer + latency + cost live.

Run locally:   streamlit run app.py
Deployed on Render as the container's start command (see render.yaml).
"""

import time
import streamlit as st

from src.indexer import LoadedIndexes
from src.semantic_cache import SemanticCache
from src.pipeline import answer_question, VALID_VARIANTS
from src import config

st.set_page_config(page_title="RAG Benchmark Demo", layout="wide")


@st.cache_resource
def load_backend():
    """
    Loaded once per container/process, not once per request -- this is
    exactly the memory footprint measured by
    scripts/check_memory_footprint.py, now living inside Streamlit
    instead of a standalone script.
    """
    indexes = LoadedIndexes()
    cache = SemanticCache()
    return indexes, cache


st.title("Cost/Latency-Optimized RAG Benchmark")
st.caption(
    "Naive vs. Cached vs. Hybrid retrieval, over a B.Tech CSE (AI/ML) sample "
    "curriculum corpus. Every answer below is a live Groq API call -- costs "
    "and latencies shown are real, not simulated."
)

if not config.GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is not set. Set it as an environment variable before "
        "running this app (locally: `export GROQ_API_KEY=...` ; on Render: "
        "set it in the service's Environment tab)."
    )
    st.stop()

with st.spinner("Loading embedding model + FAISS index + BM25 index..."):
    indexes, cache = load_backend()

query = st.text_input("Ask a question about the curriculum:", placeholder="e.g. What is the CAP theorem?")

mode = st.radio("Mode", ["Single variant", "Compare all 3 side by side"], horizontal=True)

if mode == "Single variant":
    variant = st.selectbox("Variant", VALID_VARIANTS)
    if st.button("Run", type="primary") and query.strip():
        with st.spinner(f"Running {variant}..."):
            result = answer_question(query, variant, indexes, cache=cache)
        st.markdown("### Answer")
        st.write(result.answer)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total latency", f"{result.total_latency_s:.2f}s")
        c2.metric("Retrieval latency", f"{result.retrieval_latency_s:.3f}s")
        c3.metric("LLM latency", f"{result.llm_latency_s:.3f}s")
        c4.metric("Cost", f"${result.cost_usd:.6f}")
        if result.cache_hit is not None:
            st.info(
                f"Cache {'HIT' if result.cache_hit else 'MISS'} "
                f"(similarity to nearest cached query: {result.cache_similarity_to_nearest:.3f}, "
                f"threshold: {config.CACHE_SIMILARITY_THRESHOLD})"
            )
        if result.bm25_candidate_pool_size is not None:
            st.info(f"BM25 pre-filter narrowed to {result.bm25_candidate_pool_size} candidates "
                     f"before embedding rerank.")

else:
    if st.button("Run all 3", type="primary") and query.strip():
        cols = st.columns(3)
        for col, variant in zip(cols, VALID_VARIANTS):
            with col:
                st.markdown(f"#### {variant}")
                with st.spinner(f"Running {variant}..."):
                    result = answer_question(query, variant, indexes, cache=cache)
                st.write(result.answer)
                st.metric("Latency", f"{result.total_latency_s:.2f}s")
                st.metric("Cost", f"${result.cost_usd:.6f}")
                if result.cache_hit is not None:
                    st.caption(f"Cache: {'HIT' if result.cache_hit else 'MISS'} "
                               f"(sim={result.cache_similarity_to_nearest:.3f})")
                if result.bm25_candidate_pool_size is not None:
                    st.caption(f"BM25 pool: {result.bm25_candidate_pool_size} candidates")

st.divider()
st.caption(
    f"Model: {config.GROQ_MODEL_NAME} via Groq | "
    f"Embedding: {config.EMBEDDING_MODEL_NAME} | "
    f"Cache threshold: {config.CACHE_SIMILARITY_THRESHOLD} | "
    f"BM25 pool size: {config.BM25_CANDIDATE_POOL_SIZE}"
)
