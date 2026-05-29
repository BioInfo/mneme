# Mneme - Product Requirements Document

## Overview

**Product:** Mneme
**Version:** 1.0
**Author:** Justin Johnson
**Created:** 2025-12-19
**Status:** Draft

---

## Problem Statement

Claude Code sessions contain valuable work history stored in JSONL files, but this knowledge is currently:
1. **Unsearchable** - No semantic search across past sessions
2. **Invisible** - Claude can't access prior context without expensive manual searching
3. **Growing** - Hundreds of session files accumulate, making manual retrieval impractical
4. **Context-costly** - Putting detailed history in CLAUDE.md consumes context budget

**User Pain:**
- "What did we do with the LanceDB migration?" requires hunting through files
- Reorienting Claude on previous work wastes time and context
- CLAUDE.md grows bloated trying to maintain institutional memory

---

## Solution

A semantic search system for Claude Code session history that enables quick, natural-language recall of past work.

**Core Capability:** "What did we work on related to X?" → Returns relevant session summaries and key details.

---

## User Stories

### Primary
1. **As a user**, I want to ask Claude "what did we do with the robot last week?" and get accurate results without Claude having to search through files.

2. **As a user**, I want Claude to automatically have context on recent relevant work when I mention a project.

3. **As a user**, I want my CLAUDE.md to stay lean while still having access to full session history.

### Secondary
4. **As a user**, I want to see which sessions relate to a topic and optionally load more detail.

5. **As a user**, I want the system to work across my Mac and DGX session files.

---

## Functional Requirements

### FR1: Session Indexing
- Index all session JSONL files from `~/.claude/projects/`
- Extract and embed:
  - Session summaries (`type: "summary"`)
  - User messages (`type: "user"`)
  - Assistant text responses (`type: "assistant"`, content type `"text"`)
- Exclude: thinking blocks, tool calls, tool results, binary data
- Store: session ID, timestamp, project path, content, embedding

### FR2: Semantic Search
- Natural language query: "LanceDB migration work"
- Return: Top-k relevant chunks with session context
- Include: Session date, summary, matching content excerpts

### FR3: Mneme Skill
- Skill file: `~/apps/claude-code/skills/mneme/skill.md`
- Invoked when user asks about past work or context
- Returns formatted results with session IDs for drill-down

### FR4: Incremental Updates
- Track indexed sessions (by file path + mtime)
- Only process new/modified sessions
- Run nightly alongside Obsidian indexer (or on-demand)

### FR5: Cross-Machine Support
- Index Mac sessions: `~/.claude/projects/`
- Index DGX sessions: `~/mounts/dgx/.claude/projects/`
- Unified search across both

---

## Non-Functional Requirements

### NFR1: Performance
- Search latency: < 2 seconds
- Index update: < 5 minutes for incremental
- Full reindex: < 30 minutes

### NFR2: Storage
- Vector database: LanceDB (consistent with Obsidian search)
- Location: `~/apps/claude-code/mneme/data/`
- Separate from Obsidian vectors (different schema)

### NFR3: Memory Safety
- Index on DGX if memory becomes an issue (like Obsidian indexer)
- Initial implementation: Local Mac indexing (sessions are smaller than vault)

---

## Technical Architecture

```
Session JSONL Files                    Session Vector DB
~/.claude/projects/**/*.jsonl    →    ~/apps/claude-code/mneme/data/
~/mounts/dgx/.claude/projects/         (LanceDB collection)

                    ↓
            Mneme Skill
            ~/.claude/skills/mneme/
                    ↓
            Natural Language Query
            "What did we do with vectors?"
                    ↓
            Formatted Results
            - Session date, summary
            - Relevant excerpts
            - Session ID for drill-down
```

### Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Indexer | `mneme/indexer.py` | Parse JSONL, create embeddings, store in LanceDB |
| Vector DB | `mneme/data/lance/` | LanceDB collection for session vectors |
| Search API | `mneme/search.py` | Query interface for the skill |
| Skill | `skills/mneme/skill.md` | Claude Code skill definition |
| Config | `mneme/config.yaml` | Paths, embedding model, settings |

---

## Data Model

### Session Chunk Record
```python
{
    "id": str,                    # Unique chunk ID
    "session_id": str,            # Session UUID
    "session_file": str,          # Path to JSONL file
    "project_path": str,          # e.g., "-Users-bioinfo"
    "timestamp": datetime,        # Message timestamp
    "chunk_type": str,            # "summary" | "user" | "assistant"
    "content": str,               # Text content
    "embedding": List[float],     # 768-dim vector
}
```

### Index State
```python
{
    "indexed_files": {
        "path": {"mtime": float, "chunks": int}
    },
    "last_full_index": datetime,
    "total_chunks": int,
}
```

---

## Embedding Strategy

### What to Embed

| Type | Content | Priority |
|------|---------|----------|
| `summary` | Session summary text | High - always index |
| `user` | User message content | High - captures intent |
| `assistant` (text) | Assistant responses | Medium - captures outcomes |
| `assistant` (thinking) | Thinking blocks | Skip - too verbose, internal |
| `tool_use` | Tool invocations | Skip - captured in results |
| `tool_result` | Command outputs | Skip - too large, noisy |

### Chunking Strategy
- Summaries: Index as single chunk
- User messages: Index as single chunk (usually short)
- Assistant text: Split if > 1000 chars, otherwise single chunk
- Preserve session context in metadata

### Embedding Model
- Model: `nomic-ai/nomic-embed-text-v1.5` (same as Obsidian)
- Dimension: 768
- Device: MPS (Metal) on Mac, CPU on DGX

---

## Skill Design

### Trigger Phrases
- "What did we work on..."
- "When did we..."
- "Find sessions about..."
- "Recall our work on..."
- "What was the context for..."

### Response Format
```markdown
## Mneme: [query]

### Found 3 relevant sessions:

**Dec 18, 2025** - Vector DB Migration to DGX
> Migrated Obsidian vault indexing from Mac to DGX Spark...
Session: `e8a32440-2957-4000-b830-1a3c05ed6246`

**Dec 15, 2025** - LanceDB Implementation
> Replaced FAISS with LanceDB for memory-safe incremental updates...
Session: `5dfee53c-7ad7-4e24-8f79-0d36693aa965`

**Dec 11, 2025** - Vector Search Optimization
> Implemented checkpointing and batch processing...
Session: `972e6060-ad16-451b-9364-64264dbd0c10`

---
*Use session ID to load full context if needed.*
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (MVP)
- [ ] JSONL parser with content extraction
- [ ] LanceDB schema and indexer
- [ ] Basic search function
- [ ] CLI for testing

### Phase 2: Skill Integration
- [ ] Skill definition file
- [ ] Search formatting
- [ ] Symlink to ~/.claude/skills/

### Phase 3: Automation
- [ ] Incremental index tracking
- [ ] Nightly update script
- [ ] LaunchAgent (optional)

### Phase 4: Cross-Machine
- [ ] DGX session indexing
- [ ] Unified search across sources

---

## Success Metrics

1. **Query Success Rate:** User finds relevant session on first query > 80%
2. **Latency:** Search returns in < 2 seconds
3. **Coverage:** > 95% of sessions indexed
4. **Context Savings:** CLAUDE.md stays under 150 lines

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Session files grow huge | Index slow | Incremental updates, skip unchanged |
| Embeddings too generic | Poor recall | Test and tune chunk boundaries |
| Memory issues on Mac | Crash | Move to DGX like Obsidian indexer |
| Stale index | Missing recent work | On-demand "index now" command |

---

## Open Questions

1. Should we index agent sub-sessions (agent-*.jsonl)?
2. How far back should we index? (All time vs. last N months)
3. Should the skill auto-trigger or require explicit invocation?

---

## Appendix: JSONL Format Reference

See `JSONL_FORMAT.md` for detailed analysis of session file structure.
