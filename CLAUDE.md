# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mneme is a semantic search system for Claude Code session history. It indexes JSONL session files from `~/.claude/projects/` into a LanceDB vector database, enabling natural-language queries like "what did we work on with LanceDB?"

**Status:** Phase 1 complete (core infrastructure). Phase 2 pending (skill integration).

## Architecture

```
Session JSONL Files → Indexer → LanceDB → Search API → Claude Code Skill
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

- **LanceDB** for vector storage (consistent with Obsidian search)
- **nomic-embed-text-v1.5** embeddings (768 dim, MPS acceleration)
- **Incremental indexing** via file mtime tracking
- **Content limits:** min 20 chars, max 2000 chars, batch size 100

## Documentation

- `docs/PRD.md` - Requirements and implementation phases
- `docs/TECHNICAL_DESIGN.md` - Architecture and code examples
- `docs/JSONL_FORMAT.md` - Session file format reference
