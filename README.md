# Session Recall

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![LanceDB](https://img.shields.io/badge/vector--db-LanceDB-orange.svg)](https://lancedb.com/)

**Semantic search for Claude Code session history.** Find past conversations, decisions, and context using natural language queries.

---

## The Problem

Claude Code sessions contain valuable work history stored in JSONL files, but this knowledge is:

- **Unsearchable** — No semantic search across past sessions
- **Invisible** — Claude can't access prior context without expensive manual searching
- **Growing** — Hundreds of session files accumulate over time
- **Context-costly** — Putting detailed history in CLAUDE.md consumes context budget

## The Solution

Session Recall indexes your Claude Code session files into a vector database, enabling queries like:

```
"What did we work on with LanceDB?"
"Find sessions about the authentication refactor"
"When did we discuss the API migration?"
```

---

## Features

- **Semantic Search** — Natural language queries across all your sessions
- **Session Grouping** — Results grouped by session with similarity scores
- **Incremental Indexing** — Only process new/modified sessions
- **Fast Retrieval** — Sub-second search using LanceDB
- **Skill Integration** — Works as a Claude Code skill for seamless access
- **Multi-Source Support** — Index sessions from multiple machines

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/session-recall.git
cd session-recall

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Copy and configure
cp config.example.yaml config.yaml
# Edit config.yaml to set your paths
```

### Configuration

Edit `config.yaml` to match your setup:

```yaml
sources:
  - path: "~/.claude/projects"  # Where your session files live
    name: "local"

vectordb:
  path: "./data/lance"  # Where to store the vector database

embeddings:
  device: "mps"  # Use "cpu", "cuda", or "mps" (Mac Metal)
```

### Usage

```bash
# Index your sessions (first run may take a few minutes)
python -m session_recall.cli index

# Search for past work
python -m session_recall.cli search "database migration"

# Show index statistics
python -m session_recall.cli stats
```

---

## CLI Reference

### `index` — Index session files

```bash
# Incremental index (only new/modified files)
python -m session_recall.cli index

# Full reindex
python -m session_recall.cli index --full

# Index specific source path
python -m session_recall.cli index --path ~/.claude/projects
```

### `search` — Search sessions

```bash
# Basic search
python -m session_recall.cli search "your query here"

# Limit number of sessions returned
python -m session_recall.cli search "query" --sessions 10

# Show raw chunk results instead of grouped sessions
python -m session_recall.cli search "query" --raw

# Choose a search mode
python -m session_recall.cli search "query" --mode hybrid
```

**Search modes** (`--mode` / `-m`):

| Mode | What it does | Best for |
|------|--------------|----------|
| `vector` (default) | Semantic similarity over embeddings | Conceptual recall, paraphrased queries |
| `fts` | BM25 keyword search | Exact terms, error strings, function names |
| `hybrid` | Vector + FTS fused with RRF reranking | Best non-reranked quality / low latency |
| `rerank` | Hybrid recall + local cross-encoder reorder | **Best quality** (see [benchmark](eval/BENCHMARK.md)) |

On a 27-query benchmark over a real ~138K-chunk index, `rerank` is the strongest
mode (R@1 0.815, MRR 0.864) and runs fully locally. It costs a one-time ~560MB
model download and added per-query latency. Full numbers: [eval/BENCHMARK.md](eval/BENCHMARK.md).

### `stats` — Show statistics

```bash
python -m session_recall.cli stats
```

Output:
```
Session Recall Statistics
========================================
Database path: ./data/lance
Total chunks:  15,432
Indexed files: 847
Last full index: 2025-12-19T10:30:00
```

---

## Evaluation

Search quality is easy to feel and hard to prove. The `eval/` harness measures
**recall@k** and **MRR** per search mode against a labeled query set, so changes
to the embedder, reranker, or chunking are judged by numbers — capture a
baseline, change something, re-run, compare.

```bash
./venv/bin/python eval/make_queryset.py --n 40   # template from your own sessions
# edit eval/queries.jsonl, replacing each TODO with a real query
./venv/bin/python eval/run_eval.py --save        # baseline across all modes
```

Your real query set (`eval/queries.jsonl`) is gitignored — it references your
own session IDs. See [eval/README.md](eval/README.md) for the full workflow.

---

## How It Works

### Architecture

```
Session JSONL Files → Parser → Embeddings → LanceDB → Search API
```

1. **Parse** — Extract meaningful content from session JSONL files
2. **Embed** — Generate vectors using nomic-embed-text-v1.5 (768 dimensions)
3. **Store** — Save to LanceDB with session metadata
4. **Search** — Query with natural language, retrieve by cosine similarity

### What Gets Indexed

| Content Type | Indexed | Notes |
|--------------|---------|-------|
| Session summaries | Yes | Highest value content |
| User messages | Yes | Captures intent and questions |
| Assistant text | Yes | Captures decisions and outcomes |
| Thinking blocks | No | Internal reasoning, too verbose |
| Tool calls/results | No | Technical noise |

### Data Model

Each indexed chunk contains:

```python
{
    "id": "unique-chunk-id",
    "session_id": "session-uuid",
    "session_file": "/path/to/session.jsonl",
    "project_path": "project-directory",
    "timestamp": "2025-12-19T10:30:00",
    "chunk_type": "summary|user|assistant",
    "content": "The actual text content...",
    "embedding": [0.1, 0.2, ...]  # 768-dim vector
}
```

---

## Claude Code Skill Integration

Session Recall can be used as a Claude Code skill for seamless integration:

### Skill Setup

1. Create the skill file at `~/.claude/skills/session-recall/skill.md`
2. Define trigger phrases like "recall our work on..." or "find sessions about..."
3. The skill invokes the CLI and formats results for Claude

### Example Skill Triggers

- "What did we work on with [topic]?"
- "When did we discuss [topic]?"
- "Find sessions about [topic]"
- "Recall our work on [project]"

---

## Requirements

- **Python 3.10+**
- **~2GB disk space** for embedding model (downloaded on first run)
- **Session files** from Claude Code in `~/.claude/projects/`

### Dependencies

- `lancedb` — Vector database
- `sentence-transformers` — Embedding generation
- `torch` — ML backend
- `pyarrow` — Data serialization
- `pyyaml` — Configuration
- `tqdm` — Progress bars

---

## Performance

- **Index speed**: ~1,000 sessions/minute (incremental)
- **Search latency**: <500ms for most queries
- **Storage**: ~1KB per indexed chunk

---

## Troubleshooting

### First run is slow

The embedding model (~400MB) downloads on first run. Subsequent runs are fast.

### Out of memory

Reduce `batch_size` in config.yaml:

```yaml
indexing:
  batch_size: 50  # Default is 100
```

### No results found

1. Check that sessions exist: `ls ~/.claude/projects/`
2. Run a full reindex: `python -m session_recall.cli index --full`
3. Verify index stats: `python -m session_recall.cli stats`

---

## Project Structure

```
session-recall/
├── session_recall/
│   ├── __init__.py      # Package init
│   ├── cli.py           # Command-line interface
│   ├── config.py        # Configuration loader
│   ├── parser.py        # JSONL parsing
│   ├── embeddings.py    # Vector generation
│   ├── indexer.py       # LanceDB storage
│   └── search.py        # Query interface
├── docs/
│   ├── PRD.md           # Product requirements
│   ├── TECHNICAL_DESIGN.md
│   └── JSONL_FORMAT.md  # Session file format
├── config.example.yaml  # Example configuration
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [LanceDB](https://lancedb.com/) for the excellent vector database
- [Nomic AI](https://www.nomic.ai/) for the embedding model
- [Claude Code](https://claude.ai/code) for the session format this tool indexes
