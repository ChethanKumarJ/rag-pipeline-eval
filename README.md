# rag-pipeline-eval

Production-ish RAG pipeline for doc QA with actual evals, not just a demo.

Built this after getting burned by a RAG demo that looked great in a notebook but fell apart on real docs. Wanted something I could actually measure.

## Quick start

```bash
pip install -e .
python src/ingest.py --docs ./data --db ./index
python src/app.py  # API on :8000
```

Put some PDFs / .md files in `./data` first. Then:

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "how does chunking work?"}'
```

## How it works

- `ingest.py` chunks docs (512 / 50 overlap) and stores in Chroma
- `retriever.py` does top-k retrieval + `gpt-4o-mini` for answering
- `app.py` FastAPI wrapper with `/ask` and `/health`
- `eval.py` runs RAGAS faithfulness + answer relevancy on a golden set

## Eval results

Ran on 20 hand-written QAs from my own docs:

- faithfulness: 0.82
- answer_relevancy: 0.89

Chunk size 512 beat 1024 on faithfulness (0.82 vs 0.74). Honestly surprised me, I expected larger chunks to win.

To reproduce:
```bash
python -m src.eval --golden tests/golden.json --db ./index
```

## What I'd do next

- [ ] re-ranking with cross-encoder
- [ ] hybrid search (BM25 + dense) - right now it's dense only
- [ ] caching for repeated queries
- [ ] better PDF parsing - `DirectoryLoader` is pretty basic

## Limitations

Don't use this as-is in prod - no auth, no rate limiting, eval set is tiny and biased toward my docs.
