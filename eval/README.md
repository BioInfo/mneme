# Evaluation Harness

Measure retrieval quality so changes to the embedder, reranker, or chunking are
judged by numbers instead of guesses. Reports **recall@k** and **MRR** for each
search mode (`vector`, `fts`, `hybrid`).

## Why

Semantic search quality is easy to *feel* and hard to *prove*. Before changing
the embedding model or adding a reranker, capture a baseline; after the change,
re-run and compare. If the number didn't move, the change didn't help.

## The query set

A query set is a JSONL file. Each line maps a natural-language query to the
session that *should* be retrieved:

```json
{"query": "what did we do with the vector db migration", "session_id": "abcd1234-..."}
```

Your real query set lives at `eval/queries.jsonl` and is **gitignored** — it
references your own session IDs and topics. Only the harness and a synthetic
example ship in the repo.

## Workflow

```bash
# 1. Generate a template from your own most-recent sessions
./venv/bin/python eval/make_queryset.py --n 40

# 2. Edit eval/queries.jsonl — replace each "TODO" with a query you'd really type

# 3. Capture a baseline before any change
./venv/bin/python eval/run_eval.py --save

# 4. Make a change (swap the embedder, add a reranker, change chunking, reindex)

# 5. Re-run and compare against the saved baseline in eval/results/
./venv/bin/python eval/run_eval.py
```

## Reading the output

```
mode    R@1    R@5    R@10   MRR     n
--------------------------------------
vector  0.42   0.71   0.83   0.55    40
fts     0.38   0.65   0.74   0.50    40
hybrid  0.51   0.79   0.88   0.63    40

best by MRR: hybrid (0.63)
```

- **recall@k** — fraction of queries whose target session appears in the top *k*.
- **MRR** — mean reciprocal rank; rewards putting the right session *higher*, not
  just *somewhere* in the top *k*.

## Notes

- Build the query set from sessions you genuinely remember, so the "correct"
  answer is unambiguous. 30-50 rows spanning different topics gives a stable
  signal; fewer than ~20 is noisy.
- A query set is a snapshot of your index. Regenerate it after a large reindex
  if old sessions age out of relevance.
