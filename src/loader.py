"""
Document loading + chunking.

Concept first, since this is new territory for you: a chunk is just a
slice of a document short enough to (a) fit cheaply into an LLM prompt
alongside a few other chunks, and (b) be specific enough that its
embedding represents one coherent idea rather than an averaged-out blur
of five unrelated ideas. Overlap between consecutive chunks exists so a
sentence that falls right on a chunk boundary doesn't lose its
surrounding context in *both* neighboring chunks.

We use tiktoken for token counting (LLM context limits and Groq pricing
are both denominated in tokens, not characters or words, so counting
tokens directly -- rather than approximating from word count -- keeps
chunk sizes and cost numbers accurate).
"""

import os
import glob
from dataclasses import dataclass, field
import tiktoken

from . import config

# cl100k_base is the tokenizer used by GPT-3.5/4-era OpenAI models. Groq's
# Llama models use a different tokenizer internally, but there's no public
# tiktoken-compatible encoding for Llama 3.1, so cl100k_base is used here
# as a consistent, reproducible *proxy* for chunk sizing. It won't exactly
# match Llama's real token count -- for that, actual usage numbers reported
# by the Groq API response are used instead (see groq_client.py). This
# encoding is only used to decide chunk boundaries, not to compute cost.
#
# Loaded LAZILY (on first actual use, not at import time) on purpose:
# tiktoken.get_encoding() fetches its vocab file from a remote blob store
# on first use. Loading it eagerly at module import time would mean
# simply `import`-ing this file requires network access, even for code
# that never calls chunk_text() -- e.g. importing pipeline.py to run the
# Streamlit app would transitively fail if that specific network call
# ever hiccups, for a reason that has nothing to do with what the app is
# actually doing at that moment. Lazy loading confines the network
# dependency to the one function that actually needs it.
_ENCODING = None


def _get_encoding():
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str          # source filename, e.g. "machine_learning.txt"
    text: str
    chunk_index: int      # position of this chunk within its source doc
    token_count: int = field(default=0)


def load_documents(data_dir: str = config.DATA_DIR) -> dict:
    """Load every .txt file in data_dir. Returns {filename: raw_text}."""
    docs = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            docs[filename] = f.read()
    if not docs:
        raise FileNotFoundError(
            f"No .txt files found in {data_dir}. Did you run this from "
            f"the project root, and does data/ actually have documents in it?"
        )
    return docs


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size_tokens: int = config.CHUNK_SIZE_TOKENS,
    overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS,
) -> list:
    """
    Split `text` into overlapping chunks of ~chunk_size_tokens tokens each.

    Works at the token level (encode -> slice -> decode) rather than the
    character or word level, so chunk sizes are accurate against the same
    unit the LLM and pricing are measured in.
    """
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")

    encoding = _get_encoding()
    token_ids = encoding.encode(text)
    chunks = []
    start = 0
    idx = 0
    stride = chunk_size_tokens - overlap_tokens

    while start < len(token_ids):
        end = min(start + chunk_size_tokens, len(token_ids))
        piece_ids = token_ids[start:end]
        piece_text = encoding.decode(piece_ids).strip()
        if piece_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::chunk{idx}",
                    doc_id=doc_id,
                    text=piece_text,
                    chunk_index=idx,
                    token_count=len(piece_ids),
                )
            )
            idx += 1
        if end == len(token_ids):
            break
        start += stride

    return chunks


def load_and_chunk_all(data_dir: str = config.DATA_DIR) -> list:
    """Load every document in data_dir and chunk it. Returns list[Chunk]."""
    docs = load_documents(data_dir)
    all_chunks = []
    for doc_id, text in docs.items():
        all_chunks.extend(chunk_text(text, doc_id))
    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk_all()
    print(f"Loaded and chunked {len(chunks)} chunks from data/")
    doc_counts = {}
    for c in chunks:
        doc_counts[c.doc_id] = doc_counts.get(c.doc_id, 0) + 1
    for doc_id, count in sorted(doc_counts.items()):
        print(f"  {doc_id}: {count} chunks")
