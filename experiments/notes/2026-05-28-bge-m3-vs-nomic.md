# bge-m3 vs nomic-embed-text-v1.5 — keep nomic

**Date:** 2026-05-28
**Corpus:** 9,145 session JSONL files / 125,062 chunks
**Queryset:** `eval/queries.jsonl` (n=27, hand-curated — not tracked in repo; see `eval/queries.example.jsonl` for the format)
**Result file:** `eval/results/20260528T190351Z-bge-m3.json` (gitignored)
**Baseline:** `eval/results/20260526T191407Z.json` (nomic, gitignored)

## Verdict

Keep nomic-embed-text-v1.5. Do not swap to bge-m3.

The reranker compresses most of the embedder gap, but at every other point in the search ladder bge-m3 loses, and at rerank it ties on recall and loses on MRR. There is no mode where bge-m3 wins by enough to justify the 4× model size, 1024-dim vs 768-dim storage, and the wall-clock cost of re-indexing.

## Head-to-head

| Mode | bge-m3 R@1 | nomic R@1 | bge-m3 R@5 | nomic R@5 | bge-m3 R@10 | nomic R@10 | bge-m3 MRR | nomic MRR |
|------|-----------:|----------:|-----------:|----------:|------------:|-----------:|-----------:|----------:|
| vector | 0.556 | **0.741** | 0.778 | **0.889** | 0.778 | **0.889** | 0.651 | **0.802** |
| fts | 0.593 | **0.630** | 0.852 | 0.852 | 0.852 | **0.889** | 0.691 | **0.720** |
| hybrid | 0.667 | **0.704** | 0.815 | **0.889** | 0.889 | **0.926** | 0.722 | **0.788** |
| **rerank** | 0.778 | **0.815** | 0.926 | 0.926 | 0.926 | 0.926 | 0.846 | **0.864** |

Deltas (candidate − baseline, negative = bge-m3 worse):

| Mode | dR@1 | dR@5 | dR@10 | dMRR |
|------|------:|------:|-------:|------:|
| vector | −0.185 | −0.111 | −0.111 | −0.151 |
| fts | −0.037 | 0.000 | −0.037 | −0.029 |
| hybrid | −0.037 | −0.074 | −0.037 | −0.066 |
| rerank | −0.037 | 0.000 | 0.000 | −0.018 |

## Interpretation

- **Vector-only is where embedder quality matters most**, and nomic is dramatically ahead (+0.151 MRR, +0.111 R@5). bge-m3's multilingual training mass appears not to help on this English-only technical corpus.
- **FTS is keyword, not embedder.** The small fts delta reflects chunking/tokenization differences between the two indexing passes, not an embedder property. Both runs are within noise of each other on FTS — confirms the indexing pipeline itself is stable across embedders.
- **At rerank, recall is saturated (0.926).** Both pipelines miss the same two queries. Same misses across two different embedders + the same reranker = queryset/chunking issue, not embedder issue.
- The reranker (`BAAI/bge-reranker-v2-m3`) is doing the heavy lifting. A weaker embedder costs you at vector-only (where the reranker doesn't get invoked) but the rerank ceiling is the same.

## What this rules out

- "Bigger embedder = better." Not on this corpus. bge-m3 has 568M params vs nomic's 137M, but the rerank ceiling is identical and the cheap modes are worse.
- "Multilingual is free upside." It isn't here — multilingual training mass appears to dilute English technical recall.

## What this points at

Both rerankers stably miss the same two queries. The next experiment to run is queryset-side, not embedder-side:

- One query refers to recurring work whose actual session content fragments across many sessions under varying terminology. Chunk-level rerank can't recover a query that doesn't lexically anchor.
- The other refers to a phrase that may not appear in transcripts at all — the work happened under different framing.

Action: audit `eval/queries.jsonl` for queries whose intended target doesn't lexically appear in any session transcript. Either rewrite the query to match observed phrasing or drop it as an unfindable query.

Other paths to consider (lower priority than the queryset audit):

- Smaller embedder (e.g. mxbai-embed-large, qwen3-embed-0.6b) for storage savings if recall holds.
- Hierarchical / parent-document retrieval to address the fragmented-mention failure mode directly.

## Run cost

- **First attempt: 36 GB Apple Silicon machine.** Indexing completed in ~2h 36m on MPS. Eval phase blew past physical RAM (one Python process at ~47 GB resident), saturated the macOS memory compressor (100% segment limit, 39 swapfiles), and tripped a kernel watchdog panic. Lesson: full-corpus A/B eval over 125k chunks doesn't fit alongside a normal desktop workload on 36 GB.
- **Second attempt: 119 GB unified-memory box, CPU only.** Same `embedder_ab.py` script with `--device cpu`. 10h 18m wall clock. Throughput recovered to 4–5 s/file after a competing 24 GB process was killed mid-run (memory-bandwidth contention, not OOM).
- Operational rule: full-corpus embedder A/B over this size of session history runs on the bigger box, not the laptop.

## Reproducing

`embedder_ab.py` is self-contained and reads from a base config you pass with `--base-config`. To run on any machine:

```bash
# point a base config at your source dir (or use the live config.yaml)
./venv/bin/python experiments/embedder_ab.py \
    --model "BAAI/bge-m3" --dim 1024 --label "bge-m3" \
    --doc-prefix "" --query-prefix "" \
    --device cpu --base-config path/to/base-config.yaml
```

Set `--device mps` on Apple Silicon, `--device cuda` if you have a working CUDA-enabled torch. The isolated index lands in `data/ab-<label>/`, results in `eval/results/<stamp>-<label>.json`. The script prints a delta vs the newest live-config baseline in the same dir.
