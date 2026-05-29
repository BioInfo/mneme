# Mneme

**Semantic search for Claude Code session history.** Find past conversations, decisions, and code by what you remember, not by which directory you ran them in.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![LanceDB](https://img.shields.io/badge/vector--db-LanceDB-orange.svg)](https://lancedb.com/)

---

## What it is

Claude Code stores every session as a JSONL file under `~/.claude/projects/`. After a few months you have hundreds of these — and `grep` over them is useless because you remember *what* you were doing, not the exact phrasing. Mneme indexes those JSONL files into a local LanceDB vector store and gives you four search modes (vector, full-text, hybrid, rerank) over the whole history.

A typical query: *"that time we debugged the LiteLLM gateway returning 401 for direct-API routes"*. Mneme will surface the right session even if the session itself never said "LiteLLM gateway" in those words.

This is a single-user local tool. No cloud, no API keys, no daemon. Index on demand, search from a CLI or a Claude Code skill.

## Quick start

```bash
git clone https://github.com/BioInfo/mneme.git
cd mneme

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp config.example.yaml config.yaml
# edit config.yaml — set the path to your Claude Code projects dir and the
# embedder/reranker you want. Defaults are fine for most setups.

./venv/bin/python -m mneme.cli index            # first run pulls model weights + indexes everything
./venv/bin/python -m mneme.cli search "that database migration we did"
```

The first index takes a while (model download + embedding generation over your whole history). Subsequent runs are incremental — only files changed since the last index get re-embedded.

## How it works

```
Session JSONL files (~/.claude/projects/)
        │
        ▼
   Parser            extract message text, skip tool-result noise + thinking blocks
        │
        ▼
   Chunker           split into 20–2000 char chunks, attach session metadata
        │
        ▼
   Embedder          nomic-embed-text-v1.5 (768-dim, MPS/CUDA/CPU)
        │
        ▼
   LanceDB           local vector store + FTS index
        │
        ▼
   Search ladder ───►  vector  →  fts  →  hybrid  →  rerank (bge-reranker-v2-m3)
```

Four search modes, increasing in quality and cost:

| Mode | Method | When to use |
|------|--------|-------------|
| `vector` | dense cosine | cheapest; semantic-only |
| `fts` | LanceDB BM25 | exact terms you remember |
| `hybrid` | dense + sparse fused | best default for everyday queries |
| `rerank` | hybrid candidates re-scored by a cross-encoder | when you need the top hit to be right |

The rerank mode adds a few seconds and a model download, but it's the difference between "the answer is somewhere in the top 5" and "the answer is at position 1". The eval below quantifies that.

## Eval methodology

Embedder choice, chunking, reranker, hybrid weights — all of them get opinions. Mneme ships a harness so you can settle the opinion with numbers.

**The principle:** hand-curate a queryset of ~30 queries phrased the way *you* would actually search, label each with the session ID you'd want returned, then score every retrieval mode on Recall@k and MRR. Re-run after any change. The number, not the vibes, decides.

```bash
# 1. Generate a template from your actual sessions
./venv/bin/python eval/make_queryset.py > eval/queries.jsonl

# 2. Edit eval/queries.jsonl — replace each placeholder with a real query
#    you'd type. Keep the session_id from the template (it's a real session).

# 3. Run the eval
./venv/bin/python eval/run_eval.py

# Output: eval/results/<timestamp>.json with per-mode R@1, R@5, R@10, MRR, and
# the list of queries each mode missed. Compare to the previous result to see
# whether your change helped, hurt, or did nothing.
```

See [`eval/README.md`](eval/README.md) for the full workflow and [`eval/queries.example.jsonl`](eval/queries.example.jsonl) for the format. An example results file is at [`eval/results.example.json`](eval/results.example.json).

## Performance

Numbers from a real corpus (9,145 session files / 125,062 chunks, 27 hand-curated queries):

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR |
|------|---------:|---------:|----------:|----:|
| vector (nomic-embed-text-v1.5) | 0.741 | 0.889 | 0.889 | 0.802 |
| fts | 0.630 | 0.852 | 0.889 | 0.720 |
| hybrid | 0.704 | 0.889 | 0.926 | 0.788 |
| **rerank (bge-reranker-v2-m3)** | **0.815** | **0.926** | **0.926** | **0.864** |

`rerank` is the production-grade default. 92.6% of the time the right session is in the top 5; 81.5% of the time it's at position 1.

### Embedder A/B: bge-m3 vs nomic-embed-text-v1.5

Result write-up: [`experiments/notes/2026-05-28-bge-m3-vs-nomic.md`](experiments/notes/2026-05-28-bge-m3-vs-nomic.md).

Short version: bge-m3 is 4× larger (568M vs 137M params), 1024-dim vs 768-dim storage, and loses at rerank by 0.018 MRR while tying on recall. The reranker saturates around 0.926 R@5 with either embedder, so a more expensive embedder buys nothing at the top of the ladder and is worse at vector-only.

**Keep nomic-embed-text-v1.5.** This is the kind of decision the eval harness exists for — without it, you'd reach for the bigger model and pay forever for a worse result.

## What doesn't work yet

- **Fragmented-mention queries.** When the same recurring topic gets discussed under 5 different framings across 20 sessions, chunk-level rerank can't recover a query that doesn't lexically anchor to any one chunk. Both embedders we've tested miss the same queries here. The fix is probably hierarchical retrieval (session-level summaries above chunk-level vectors), not a different embedder.
- **Index portability.** The LanceDB files are local. No sync story across machines yet. If you want session search on a second box, you re-index there.
- **Schema migration.** If you change the embedder dimension (e.g., 768 → 1024), you have to `index --full`. The harness in `experiments/embedder_ab.py` does this cleanly in an isolated directory so the live index isn't touched.

## CLI reference

```bash
# Index
./venv/bin/python -m mneme.cli index                  # incremental (default)
./venv/bin/python -m mneme.cli index --full           # full reindex
./venv/bin/python -m mneme.cli index --path PATH      # specific source

# Search
./venv/bin/python -m mneme.cli search "query"                   # rerank, top 5 sessions
./venv/bin/python -m mneme.cli search "query" --sessions 10     # more sessions
./venv/bin/python -m mneme.cli search "query" --mode hybrid     # cheaper, faster
./venv/bin/python -m mneme.cli search "query" --raw             # chunks, not grouped sessions

# Stats
./venv/bin/python -m mneme.cli stats
```

## Claude Code skill integration

Drop in a skill that wraps the CLI:

```bash
mkdir -p ~/.claude/skills/mneme
```

`~/.claude/skills/mneme/SKILL.md`:

```markdown
---
name: mneme
description: Search past Claude Code sessions semantically. Use when the user
  asks about previous work, past sessions, what was discussed before, or needs
  to recall context from earlier conversations.
---

When the user wants to recall past work, run:

    bash ~/.claude/skills/mneme/search.sh "<their query>"

Return the top sessions verbatim.
```

`~/.claude/skills/mneme/search.sh`:

```bash
#!/bin/bash
cd /path/to/mneme && ./venv/bin/python -m mneme.cli search "$1" --sessions "${2:-5}" --mode rerank
```

Triggers in conversation: "what did we work on with X", "find the session where Y", "remind me about Z".

## Requirements

- Python 3.10+
- ~2 GB disk for model cache (sentence-transformers downloads on first run)
- ~1 GB per ~10k indexed chunks for the LanceDB store
- Apple Silicon (MPS) or CUDA recommended for indexing speed; CPU works fine for search

### Dependencies

`lancedb`, `sentence-transformers`, `pyyaml`, `tqdm`. See `requirements.txt`.

## Project structure

```
mneme/
├── mneme/                  # Python package
│   ├── cli.py              # Command-line entry point
│   ├── parser.py           # JSONL parsing
│   ├── embeddings.py       # sentence-transformers wrapper
│   ├── indexer.py          # LanceDB write path
│   ├── search.py           # Query interface
│   ├── rerank.py           # Cross-encoder reranking
│   └── config.py           # YAML config + path expansion
├── eval/                   # Eval harness — see eval/README.md
│   ├── make_queryset.py    # Generate a queryset template
│   ├── run_eval.py         # Run + score
│   ├── queries.example.jsonl
│   ├── results.example.json
│   ├── README.md
│   └── BENCHMARK.md        # Latest numbers
├── experiments/            # A/B harness for swappable components
│   ├── embedder_ab.py      # Apples-to-apples embedder comparison
│   └── notes/              # Write-ups, e.g. bge-m3 vs nomic
├── docs/
│   ├── PRD.md              # Requirements
│   ├── TECHNICAL_DESIGN.md # Architecture
│   └── JSONL_FORMAT.md     # Session file format reference
├── config.example.yaml     # Copy to config.yaml and edit
├── requirements.txt
└── README.md
```

## Contributing

PRs welcome. Two house rules:

1. **No claim without an eval number.** "X is better" gets paired with a `eval/results/` file showing the delta. The harness is set up to make this cheap.
2. **No private data in the queryset.** `eval/queries.jsonl` is gitignored for a reason — your queries reference your sessions. If you want to share a benchmark, generate a queryset that doesn't reveal what you've been working on.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

- [LanceDB](https://lancedb.com/) for the local vector + FTS store.
- [Nomic](https://nomic.ai/) for `nomic-embed-text-v1.5`.
- [BAAI](https://www.baai.ac.cn/) for `bge-reranker-v2-m3`.
