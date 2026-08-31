"""
Central configuration for the RAG benchmark.

Everything that's a "tunable knob" lives here, in one place, so you can
change it without hunting through files. Every value below is a concrete
decision, not a placeholder -- if you change one, update the comment
explaining why the original value was chosen.
"""

import os

# --------------------------------------------------------------------------
# Embedding model
# --------------------------------------------------------------------------
# all-MiniLM-L6-v2: 384-dim embeddings, ~90MB on disk, runs fast on CPU.
# This is the standard "good enough, cheap, local" sentence-transformers
# model -- it's what most RAG tutorials default to for a reason: it's small
# enough to not blow Render's free-tier memory ceiling, and accurate enough
# for a portfolio-scale corpus like this one.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --------------------------------------------------------------------------
# Groq LLM
# --------------------------------------------------------------------------
# Llama 3.1 8B Instant: $0.05 / 1M input tokens, $0.08 / 1M output tokens.
# This is Groq's cheapest production text model as of Aug 2026 (verified via
# web search at build time -- see README for the pricing source). Picked
# specifically because the whole point of this project is cost tracking,
# and a cheap model keeps total spend trivial while iterating.
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Pricing is in USD per token (not per million) so cost math elsewhere
# doesn't need to remember to divide by 1e6 every time.
GROQ_PRICE_PER_INPUT_TOKEN = 0.05 / 1_000_000
GROQ_PRICE_PER_OUTPUT_TOKEN = 0.08 / 1_000_000

# If Groq's pricing changes, update the three constants above and this date.
GROQ_PRICING_LAST_VERIFIED = "2026-08-31"

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50  # ~10% overlap, standard default to avoid
# cutting a sentence/idea in half at a chunk boundary and losing it from
# both neighboring chunks' context.

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
TOP_K = 4  # number of chunks sent to the LLM as context

# --------------------------------------------------------------------------
# Hybrid retrieval (BM25 pre-filter -> embedding rerank)
# --------------------------------------------------------------------------
# Concrete number, chosen and logged explicitly per your instruction not to
# leave it implicit. 20-30 chunks is small enough that the embedding rerank
# step is cheap (it's re-embedding only the query, then just doing a
# vector-similarity comparison against a small candidate set -- not
# re-embedding documents), but large enough that BM25 is unlikely to have
# already excluded the actually-relevant chunk for most queries in a corpus
# this size (~20 docs x a handful of chunks each).
BM25_CANDIDATE_POOL_SIZE = 25

# --------------------------------------------------------------------------
# Semantic cache
# --------------------------------------------------------------------------
# Starting point per your spec -- treat as tunable. The pipeline logs the
# actual nearest-neighbor similarity score for every query (hit or miss) to
# results.csv/SQLite specifically so this threshold can be tuned later
# against real data instead of guessed twice.
CACHE_SIMILARITY_THRESHOLD = 0.92

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
QUESTIONS_PATH = os.path.join(BASE_DIR, "questions.json")
INDEX_DIR = os.path.join(BASE_DIR, "index_store")  # committed to repo,
# built once by scripts/build_index.py, then loaded (not rebuilt) at
# app/container startup -- see README for the Render free-tier reasoning.
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_METADATA_PATH = os.path.join(INDEX_DIR, "chunks.json")
BM25_INDEX_PATH = os.path.join(INDEX_DIR, "bm25.pkl")

RESULTS_DB_PATH = os.path.join(BASE_DIR, "results.db")
RESULTS_CSV_PATH = os.path.join(BASE_DIR, "results.csv")
CACHE_DB_PATH = os.path.join(BASE_DIR, "cache_store.db")  # bundled local
# SQLite file for the semantic cache -- no external Redis, per your
# constraint, and it survives within a single Render container instance.
