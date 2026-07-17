# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Read first at session start:** [`.claude/CONTINUITY.md`](.claude/CONTINUITY.md) — tactical session handoff; update at session end.

## Project Overview

Mneme is a semantic search system for Claude Code session history, covering the
whole fleet. It indexes JSONL session files from mac, dgx, pi and mini into a
LanceDB vector database on the DGX, and answers natural-language queries like
"what did we work on with LanceDB?" from any machine.

**Status:** running. One index on the DGX (52,611 files), served over HTTP on
:6336. The mac, mini and Pi hold no engine and no database; the skill is a thin
client. Every chunk carries the host it came from.

**Read `.claude/CONTINUITY.md` first** — it holds the live cursor, the open
decisions, and the reliability debt.

## Architecture

```
mac ─┐
pi  ─┼─ rsync ─▶ dgx:~/session-sources/<host>/ ─▶ sanitize ─▶ Indexer ─▶ LanceDB
mini┘                                                                      │
dgx ─── local ─────────────────────────────────────────────────────────────┤
                                                        mneme_server.py :6336
                                                                           │
                                              skill on any host ──HTTP─────┘
```

### Core Components

| File | Purpose | Status |
|------|---------|--------|
| `parser.py` | Parse JSONL, extract summaries/user/assistant text | Done |
| `embeddings.py` | Generate vectors using nomic-embed-text-v1.5 | Done |
| `indexer.py` | LanceDB storage with SessionVectorDB class | Done |
| `search.py` | Query interface with result formatting | Done |
| `cli.py` | Command-line: `index`, `search`, `stats` | Done |
| `config.py` | YAML config loader with path expansion | Done |

### Data Flow

1. **Index:** Parse JSONL → Extract content → Embed → Store in LanceDB
2. **Search:** Embed query → Cosine similarity → Group by session → Format

## Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate  # or: source venv/bin/activate.fish
pip install -r requirements.txt

# CLI usage
./venv/bin/python -m mneme.cli index          # Incremental index
./venv/bin/python -m mneme.cli index --full   # Full reindex
./venv/bin/python -m mneme.cli search "query" # Search sessions
./venv/bin/python -m mneme.cli stats          # Show index stats
```

## JSONL Parsing Rules

**Index these:**
- `type: "summary"` → `record['summary']` (highest value)
- `type: "user"` → `record['message']['content']` (if string, not tool_result)
- `type: "assistant"` → `block['text']` where `block['type'] == 'text'`

**Skip these:**
- `type: "file-history-snapshot"` (no semantic value)
- `type: "assistant"` with `thinking` or `tool_use` blocks
- `type: "user"` with tool_result arrays

## Key Design Decisions

- **LanceDB** for vector storage. This used to say "consistent with Obsidian
  search", which was true in Dec 2025 and became false at the vault's May cutover
  to Qdrant. The two stacks are separate by accident, not design. **Decided
  2026-07-17: migrate sessions onto a Qdrant collection** (768d/nomic, alongside
  `obsidian-v3` at 4096d/qwen3 — Qdrant allows per-collection dims), then delete
  this API and its unit. See `.claude/CONTINUITY.md`. Do not re-embed with qwen3.
- **nomic-embed-text-v1.5** embeddings (768 dim). CUDA on the DGX, measured 17.4x
  over the CPU wheel. `device` is unset in config so it auto-detects mps > cuda > cpu.
- **Incremental indexing** via file mtime tracking
- **Content limits:** min 20 chars, max 2000 chars
- **`batch_size: 2000`, and it must stay large.** It is not just an embedder batch:
  it is one LanceDB append per cycle, and each append writes a manifest listing
  every prior fragment, so a small value is quadratic in table size. Measured at
  52,611 files: batch 100 gave 2.41 files/sec and a 5h31m ETA; batch 2000 gave
  ~29.6 files/sec and ~30min. The GPU was never the bottleneck.
- **A required source that is missing or empty is a hard failure** (`SourceError`,
  exit 1). It used to print a warning and exit 0, which is how the mac indexed zero
  DGX sessions for months without an alert.

## Documentation

- `docs/PRD.md` - Requirements and implementation phases
- `docs/TECHNICAL_DESIGN.md` - Architecture and code examples
- `docs/JSONL_FORMAT.md` - Session file format reference
