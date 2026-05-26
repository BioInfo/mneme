# Benchmark

Measured retrieval quality of each search mode, run with the `eval/` harness
against a real Claude Code session index. Reproduce on your own index by
building a query set (`make_queryset.py`) and running `run_eval.py`.

## Setup

- **Index:** ~138K chunks across ~11K real Claude Code session files
- **Query set:** 27 hand-written queries, each phrased as the kind of question a
  user would later ask (intent, not echoing the indexed text), mapped to the
  one session that should be retrieved
- **Embedder:** `nomic-ai/nomic-embed-text-v1.5` (768-dim), local on Apple MPS
- **Reranker:** `BAAI/bge-reranker-v2-m3`, local
- **k:** 10 (retrieval depth)
- **Hardware:** Apple Silicon (M-series), fully local — no API, no remote index

## Results

| mode | R@1 | R@5 | R@10 | MRR |
|--------|-------|-------|-------|-------|
| vector | 0.741 | 0.889 | 0.889 | 0.802 |
| fts    | 0.630 | 0.852 | 0.889 | 0.720 |
| hybrid | 0.704 | 0.889 | 0.926 | 0.788 |
| **rerank** | **0.815** | **0.926** | **0.926** | **0.864** |

## Reading it

- **hybrid** has the best first-stage recall (R@10 0.926) but ranks the target
  worse than plain vector (R@1 0.704 vs 0.741) — it finds the right session but
  buries it.
- **rerank** runs a local cross-encoder over hybrid's candidates and reorders
  them. It keeps the 0.926 recall ceiling and converts it into precision:
  **R@1 +0.111 and MRR +0.076 over hybrid**, the best of every mode. The right
  session is the top hit 81.5% of the time.
- The reranker cannot beat the recall of its candidate pool. The 2 queries it
  misses are ones where the target session never enters the top candidates —
  a first-stage (embedder) limitation, not a reranking one. Upgrading the
  embedder is the lever for those.

## Takeaway

For quality, `--mode rerank` wins and runs fully locally. The cost is a one-time
~560MB model download and added per-query latency from the cross-encoder. For
lowest latency with no extra model, `hybrid` is the best non-reranked mode.
