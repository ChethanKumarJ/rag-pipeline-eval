# rag-pipeline-eval

Production-ish RAG pipeline for doc QA with actual evals, not just a demo.

Built this after getting burned by a RAG demo that looked great in a notebook but fell apart on real docs. Wanted something I could actually measure.

## Quick start

```bash
pip install -e .
python src/ingest.py --docs ./data --db ./index
python src/app.py  # API on :8000
```

Put some PDFs / .md files in `./data` first.

## How it works

- `ingest.py` chunks docs and stores in Chroma
- `retriever.py` does hybrid-ish retrieval (dense for now, BM25 TODO)
- `app.py` FastAPI wrapper with `/ask`
- `eval.py` runs RAGAS faithfulness + answer relevancy on a small golden set

See `tests/` for the eval set format.

## What I learned

Chunk size matters way more than embedding model for my test docs. 512 tokens with 50 overlap beat 1024 on faithfulness.

Still TODO: re-ranking, proper hybrid search, caching.
