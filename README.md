# Cost/Latency-Optimized RAG Benchmark

A portfolio project comparing three implementations of the same RAG (Retrieval-Augmented Generation) pipeline — **naive**, **cached**, and **hybrid retrieval** — on cost, latency, and accuracy over the same document set and question set. The point isn't a chat-with-your-PDF demo; it's demonstrating the engineering judgment behind *why* you'd pick one retrieval strategy over another in production, backed by measured numbers instead of intuition.

## What's actually being compared

| Variant | What it does | What it optimizes for |
|---|---|---|
| **Naive RAG** | Embed query → dense vector search over the whole corpus → send top-k chunks to the LLM → return answer | Baseline. Simple, always correct-as-possible, always pays full cost. |
| **Cached RAG** | Naive RAG + a semantic cache in front of it. If a semantically similar question was asked before (cosine similarity above a threshold), reuse that answer instead of calling the LLM again | Repeated/similar traffic — near-zero cost and latency on a cache hit |
| **Hybrid Retrieval RAG** | Cheap BM25 keyword search first narrows the corpus to a small candidate pool, THEN the expensive embedding-based search only runs within that smaller pool | Retrieval compute cost, at scale — the pattern that makes semantic search tractable over millions of chunks |

## Corpus and question set

`data/` contains 22 short text documents covering standard B.Tech CSE (AI/ML specialization) subjects — data structures, algorithms, OS, networks, DBMS, software engineering, ML, deep learning, NLP, computer vision, information retrieval, cloud computing, cybersecurity, distributed systems, and more.

**Important honesty note:** these are *not* your actual MUJ syllabus PDFs — none were available when this was built. They're original reference notes I wrote covering the standard topics in each subject (verified against publicly available information about MUJ's CSE/AI-ML curriculum structure), sized and structured to behave like a real syllabus corpus for benchmarking purposes. If you get your actual syllabus PDFs later, drop them in `data/` (as `.txt`, or extend `src/loader.py` to handle PDFs directly — see the `pdf` skill's approach if you want a quick text-extraction script) and re-run `scripts/build_index.py`. The benchmark methodology doesn't change; only the corpus does.

`questions.json` has 26 questions: 10 exact-fact lookups, 9 conceptual questions, and **7 multi-document questions** (each deliberately written to require synthesizing information from two specific documents — e.g., relating the CAP theorem in distributed systems to ACID guarantees in DBMS). The multi-doc questions are the ones most likely to expose a real accuracy gap between naive/hybrid retrieval, since BM25's keyword-only pre-filter can miss a relevant chunk that uses different vocabulary than the query.

## Architecture

```
data/*.txt  →  loader.py (chunk)  →  indexer.py (embed + build FAISS + BM25)
                                            ↓
                                     index_store/ (committed to repo)
                                            ↓
              pipeline.py: answer_question(query, variant)
                     ↓            ↓             ↓
              naive_rag.py  cached_rag.py  hybrid_rag.py
                     ↓            ↓             ↓
                        groq_client.py (instrumented LLM calls)
                                  ↓
                        logging_db.py → results.db → results.csv
```

Every LLM call goes through `groq_client.py`, which returns latency, exact input/output token counts (from Groq's own API response, not estimated), and cost for every single call — this instrumentation is the actual deliverable of the project, not an afterthought bolted on at the end.

**On LangChain:** the brief asked for LangChain retrievers/chains "where they fit naturally." I deliberately did *not* wrap this in LangChain's `Chain`/`EnsembleRetriever` abstractions. The hybrid variant's specific mechanism — BM25 narrows the field, THEN embedding search reranks only within that narrowed subset — doesn't map cleanly onto LangChain's built-in `EnsembleRetriever` (which does independent parallel retrieval + reciprocal rank fusion, a different algorithm). Given your explicit ask to keep every step explainable without hidden abstraction, I used `sentence-transformers` + `faiss` + `rank_bm25` directly. Be ready to explain this trade-off in an interview: it's a deliberate choice, not an oversight, but LangChain's abstractions *would* still be the right call if you were building this for a real product rather than to explain every wire to an interviewer.

## Project structure

```
data/                     22 source documents
questions.json            26 test questions with type + relevant_docs labels
src/
  config.py                all tunable constants in one place
  loader.py                document loading + token-based chunking
  indexer.py                embedding + FAISS + BM25 index building/loading
  groq_client.py            instrumented Groq API wrapper
  semantic_cache.py         cosine-similarity cache, SQLite-backed
  naive_rag.py               naive variant
  cached_rag.py               cached variant
  hybrid_rag.py                hybrid variant
  pipeline.py                   answer_question(query, variant) router
  logging_db.py                  SQLite results logging + CSV export
scripts/
  build_index.py            builds index_store/ from data/ (run once)
  check_memory_footprint.py  reports RSS with everything loaded
  run_benchmark.py           runs questions.json through variant(s), logs results
  analyze_results.py         summary table + charts + RESULTS.md from graded results.csv
app.py                     Streamlit demo
requirements.txt           pinned dependencies (CPU-only torch)
render.yaml                Render deployment config
```

## Running it locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

export GROQ_API_KEY=your_key_here     # Windows: set GROQ_API_KEY=your_key_here

# 1. Build the index (downloads the embedding model on first run, ~90MB)
python scripts/build_index.py

# 2. Sanity-check memory footprint before building anything else
python scripts/check_memory_footprint.py

# 3. Run ONE variant at a time, grading between each (see "Working style" below)
python scripts/run_benchmark.py --variant naive
#   -> open results.csv, fill in the 'correctness' column for the naive rows
#      (correct / partial / wrong) by hand
python scripts/run_benchmark.py --variant cached
#   -> grade cached rows
python scripts/run_benchmark.py --variant hybrid
#   -> grade hybrid rows

# 4. Generate the summary table, charts, and RESULTS.md from your graded CSV
python scripts/analyze_results.py

# 5. Run the demo
streamlit run app.py
```

### Working style this project was built for

Each variant was built and handed over as a complete, runnable unit rather than all three being written and run blind. The intended loop:

1. Build naive RAG, run it, **grade its `results.csv` rows by hand** before building cached RAG.
2. Build cached RAG, run it, grade those rows before building hybrid.
3. Build hybrid, run it, grade those rows.
4. Only then run `analyze_results.py` for the full comparison.

This project's code was written all at once (per your choice), but the *running and grading* should still follow this sequence — `run_benchmark.py --variant X` lets you do exactly that.

## Deployment (Render)

1. Push this repo to GitHub, including the committed `index_store/` directory (it's small — built from 22 short text files).
2. On Render: New → Web Service → connect the repo. Render will read `render.yaml` automatically, or set the build/start commands manually:
   - Build: `pip install -r requirements.txt`
   - Start: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
3. Set the `GROQ_API_KEY` environment variable in Render's dashboard (Environment tab) — never commit it.
4. Deploy. First load will be slow (cold start + downloading the ~90MB embedding model if it's not cached in the image — consider baking the model into the build if cold starts are too slow for your demo).

**Cold starts are expected, not a bug.** Render's free tier spins down the service after a period of inactivity; the next request triggers a fresh container start, which takes noticeably longer than a warm request. This is a known trade-off of free hosting — the fix, if it matters for your demo, is upgrading to a paid always-on plan, not trying to prevent the spin-down.

**Memory ceiling:** run `scripts/check_memory_footprint.py` locally before deploying and compare against Render free tier's 512MB cap (see the script's own printed interpretation). If you extend the corpus or swap to a larger embedding model later, re-run this check — it's cheap insurance against a deploy that builds fine but crashes on first request.

## Results

See [`RESULTS.md`](RESULTS.md) — generated by `scripts/analyze_results.py` from your actual graded run. It is not filled in yet in this delivery, since I can't execute Groq API calls or download the embedding model from the sandbox this was built in (no network access to `api.groq.com` or `huggingface.co`). Run the steps above end-to-end once, and `analyze_results.py` will populate it with your real numbers and generate the three charts (`chart_cost.png`, `chart_latency.png`, `chart_accuracy.png`) referenced inside it.

## Known limitations / honest caveats

- The corpus is a synthetic stand-in for your real syllabus (see above) — swap it in when you have the actual PDFs.
- BM25 tokenization is simple whitespace + lowercase, no stemming — fine at this corpus size, would need revisiting at larger scale.
- The semantic cache's similarity threshold (0.92, in `config.py`) is a starting point, not a validated value — `results.csv` logs the actual nearest-neighbor similarity for every query (hit or miss) specifically so you can tune it against real data instead of guessing again.
- Accuracy grading is manual by design (per your instruction not to auto-grade with an LLM judge) — the numbers in `RESULTS.md` are only as good as how carefully you grade `results.csv`.
- This was built and code-reviewed without being executed end-to-end in this environment. Treat the first local run as the real first test — if something breaks, it's most likely a small dependency/version mismatch (see `requirements.txt`) rather than a logic error, since the pipeline wiring was checked by hand, not by a test run.
