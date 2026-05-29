# Mneme - Technical Design Document

## Overview

This document details the technical implementation of the Mneme system for semantic search across Claude Code session history.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Session Sources                          │
├─────────────────────────────────────────────────────────────────┤
│  ~/.claude/projects/**/*.jsonl      (Mac local sessions)        │
│  ~/mounts/dgx/.claude/projects/     (DGX sessions via SSHFS)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Indexer                                  │
│  ~/apps/claude-code/mneme/indexer.py                   │
├─────────────────────────────────────────────────────────────────┤
│  1. Discover session files                                       │
│  2. Parse JSONL, extract content                                 │
│  3. Chunk and embed text                                         │
│  4. Store in LanceDB                                             │
│  5. Track index state                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Vector Database                            │
│  ~/apps/claude-code/mneme/data/lance/                  │
├─────────────────────────────────────────────────────────────────┤
│  Collection: session_chunks                                      │
│  Schema: id, session_id, project, timestamp, type, content, vec │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Search Module                              │
│  ~/apps/claude-code/mneme/search.py                    │
├─────────────────────────────────────────────────────────────────┤
│  1. Embed query                                                  │
│  2. Vector similarity search                                     │
│  3. Group by session                                             │
│  4. Format results                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Mneme Skill                         │
│  ~/.claude/skills/mneme/                               │
├─────────────────────────────────────────────────────────────────┤
│  Invoked by Claude Code when user asks about past work          │
│  Calls search module, formats response                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
~/apps/claude-code/mneme/
├── PRD.md                    # Product requirements
├── TECHNICAL_DESIGN.md       # This document
├── JSONL_FORMAT.md          # Session file format reference
├── config.yaml              # Configuration
├── requirements.txt         # Python dependencies
├── indexer.py               # Main indexer script
├── search.py                # Search API
├── parser.py                # JSONL parsing utilities
├── embeddings.py            # Embedding generation
├── cli.py                   # Command-line interface
├── data/
│   ├── lance/               # LanceDB database
│   └── index_state.json     # Tracking indexed files
└── tests/
    ├── test_parser.py
    ├── test_search.py
    └── fixtures/
```

---

## Component Details

### 1. JSONL Parser (`parser.py`)

**Purpose:** Extract searchable content from session JSONL files.

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Optional
import json

@dataclass
class SessionChunk:
    session_id: str
    session_file: str
    project_path: str
    timestamp: datetime
    chunk_type: str  # "summary" | "user" | "assistant"
    content: str
    message_id: Optional[str] = None

def parse_session_file(file_path: str) -> Iterator[SessionChunk]:
    """Parse a session JSONL file and yield indexable chunks."""
    project_path = extract_project_from_path(file_path)

    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line)

            # Session summaries (high value)
            if record.get('type') == 'summary':
                yield SessionChunk(
                    session_id=record.get('leafUuid', ''),
                    session_file=file_path,
                    project_path=project_path,
                    timestamp=datetime.now(),  # Summaries don't have timestamps
                    chunk_type='summary',
                    content=record.get('summary', ''),
                )

            # User messages
            elif record.get('type') == 'user':
                msg = record.get('message', {})
                content = extract_user_content(msg.get('content', ''))
                if content and len(content) > 10:  # Skip tiny messages
                    yield SessionChunk(
                        session_id=record.get('sessionId', ''),
                        session_file=file_path,
                        project_path=project_path,
                        timestamp=parse_timestamp(record.get('timestamp')),
                        chunk_type='user',
                        content=content,
                        message_id=record.get('uuid'),
                    )

            # Assistant text responses
            elif record.get('type') == 'assistant':
                msg = record.get('message', {})
                for block in msg.get('content', []):
                    if block.get('type') == 'text':
                        text = block.get('text', '')
                        if text and len(text) > 50:  # Skip tiny responses
                            yield SessionChunk(
                                session_id=record.get('sessionId', ''),
                                session_file=file_path,
                                project_path=project_path,
                                timestamp=parse_timestamp(record.get('timestamp')),
                                chunk_type='assistant',
                                content=text[:2000],  # Truncate long responses
                                message_id=record.get('uuid'),
                            )

def extract_user_content(content) -> str:
    """Extract text from user message content (can be string or list)."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Handle tool results - skip them
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                texts.append(item.get('text', ''))
        return ' '.join(texts)
    return ''
```

### 2. Embeddings (`embeddings.py`)

**Purpose:** Generate vector embeddings for text chunks.

```python
from sentence_transformers import SentenceTransformer
from typing import List
import torch

class EmbeddingModel:
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.dimension = 768

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        # Nomic requires task prefix for better retrieval
        prefixed = [f"search_document: {t}" for t in texts]
        embeddings = self.model.encode(prefixed, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a search query."""
        prefixed = f"search_query: {query}"
        embedding = self.model.encode([prefixed], convert_to_numpy=True)
        return embedding[0].tolist()
```

### 3. Vector Database (`indexer.py`)

**Purpose:** Store and retrieve session chunks using LanceDB.

```python
import lancedb
from pathlib import Path
import pyarrow as pa

class SessionVectorDB:
    TABLE_NAME = "session_chunks"

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        self._ensure_table()

    def _ensure_table(self):
        """Create table if it doesn't exist."""
        if self.TABLE_NAME not in self.db.table_names():
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("session_id", pa.string()),
                pa.field("session_file", pa.string()),
                pa.field("project_path", pa.string()),
                pa.field("timestamp", pa.timestamp('us')),
                pa.field("chunk_type", pa.string()),
                pa.field("content", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), 768)),
            ])
            self.db.create_table(self.TABLE_NAME, schema=schema)

    def add_chunks(self, chunks: list):
        """Add chunks to the database."""
        table = self.db.open_table(self.TABLE_NAME)
        table.add(chunks)

    def search(self, query_vector: list, limit: int = 10) -> list:
        """Search for similar chunks."""
        table = self.db.open_table(self.TABLE_NAME)
        results = table.search(query_vector).limit(limit).to_list()
        return results

    def get_stats(self) -> dict:
        """Get database statistics."""
        table = self.db.open_table(self.TABLE_NAME)
        return {
            "total_chunks": table.count_rows(),
            "db_path": str(self.db_path),
        }
```

### 4. Search API (`search.py`)

**Purpose:** High-level search interface for the skill.

```python
from dataclasses import dataclass
from typing import List
from datetime import datetime

@dataclass
class SearchResult:
    session_id: str
    session_file: str
    project_path: str
    timestamp: datetime
    chunk_type: str
    content: str
    similarity: float

def search_sessions(query: str, limit: int = 10) -> List[SearchResult]:
    """Search for sessions matching the query."""
    from .embeddings import EmbeddingModel
    from .indexer import SessionVectorDB

    # Load config
    config = load_config()

    # Initialize components
    embedder = EmbeddingModel(config['embeddings']['model'])
    db = SessionVectorDB(config['vectordb']['path'])

    # Generate query embedding
    query_vector = embedder.embed_query(query)

    # Search
    results = db.search(query_vector, limit=limit)

    # Convert to SearchResult objects
    return [
        SearchResult(
            session_id=r['session_id'],
            session_file=r['session_file'],
            project_path=r['project_path'],
            timestamp=r['timestamp'],
            chunk_type=r['chunk_type'],
            content=r['content'],
            similarity=r['_distance'],
        )
        for r in results
    ]

def format_results(results: List[SearchResult]) -> str:
    """Format search results for display."""
    if not results:
        return "No matching sessions found."

    # Group by session
    sessions = {}
    for r in results:
        if r.session_id not in sessions:
            sessions[r.session_id] = {
                'timestamp': r.timestamp,
                'project': r.project_path,
                'chunks': [],
                'best_similarity': r.similarity,
            }
        sessions[r.session_id]['chunks'].append(r)

    # Format output
    lines = []
    for sid, data in sorted(sessions.items(), key=lambda x: x[1]['timestamp'], reverse=True)[:5]:
        date_str = data['timestamp'].strftime('%Y-%m-%d')
        summary = next((c.content for c in data['chunks'] if c.chunk_type == 'summary'), None)

        lines.append(f"**{date_str}** - {data['project']}")
        if summary:
            lines.append(f"> {summary[:200]}...")
        else:
            # Use first user message as summary
            user_msg = next((c.content for c in data['chunks'] if c.chunk_type == 'user'), '')
            lines.append(f"> {user_msg[:200]}...")
        lines.append(f"Session: `{sid}`")
        lines.append("")

    return '\n'.join(lines)
```

### 5. CLI (`cli.py`)

**Purpose:** Command-line interface for indexing and testing.

```python
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Mneme CLI')
    subparsers = parser.add_subparsers(dest='command')

    # Index command
    index_parser = subparsers.add_parser('index', help='Index session files')
    index_parser.add_argument('--full', action='store_true', help='Full reindex')
    index_parser.add_argument('--path', type=str, help='Specific path to index')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search sessions')
    search_parser.add_argument('query', type=str, help='Search query')
    search_parser.add_argument('--limit', type=int, default=10, help='Max results')

    # Stats command
    subparsers.add_parser('stats', help='Show index statistics')

    args = parser.parse_args()

    if args.command == 'index':
        from .indexer import run_indexer
        run_indexer(full=args.full, path=args.path)
    elif args.command == 'search':
        from .search import search_sessions, format_results
        results = search_sessions(args.query, limit=args.limit)
        print(format_results(results))
    elif args.command == 'stats':
        from .indexer import SessionVectorDB
        config = load_config()
        db = SessionVectorDB(config['vectordb']['path'])
        print(db.get_stats())

if __name__ == '__main__':
    main()
```

---

## Configuration

**`config.yaml`:**
```yaml
# Mneme Configuration

# Session file locations
sources:
  - path: "~/.claude/projects"
    name: "mac-local"
  - path: "~/mounts/dgx/.claude/projects"
    name: "dgx"
    optional: true  # Don't fail if not mounted

# Vector database
vectordb:
  path: "~/apps/claude-code/mneme/data/lance"

# Embedding model
embeddings:
  model: "nomic-ai/nomic-embed-text-v1.5"
  dimension: 768
  device: "mps"  # or "cpu" for DGX

# Indexing settings
indexing:
  batch_size: 100
  min_content_length: 20
  max_content_length: 2000
  skip_agent_sessions: false  # Include agent-*.jsonl files

# Search settings
search:
  default_limit: 10
  group_by_session: true
```

---

## Index State Tracking

**`data/index_state.json`:**
```json
{
  "version": 1,
  "last_full_index": "2025-12-19T10:00:00Z",
  "last_incremental": "2025-12-19T15:30:00Z",
  "indexed_files": {
    "~/.claude/projects/-Users-bioinfo/abc123.jsonl": {
      "mtime": 1734567890.123,
      "chunks": 45,
      "indexed_at": "2025-12-19T10:05:00Z"
    }
  },
  "stats": {
    "total_files": 412,
    "total_chunks": 15234,
    "total_sessions": 398
  }
}
```

---

## Skill Integration

**`~/apps/claude-code/skills/mneme/SKILL.md`:**
```markdown
---
name: recalling-session-history
description: Search and recall past Claude Code session history. Use when user asks about previous work, past conversations, or needs context from earlier sessions.
version: 1.0.0
---

# Mneme

Search semantic history across all Claude Code sessions to find relevant past work.

## When to Use

- User asks "what did we work on..." or "when did we..."
- User mentions a past project and needs context
- User wants to recall decisions or implementations from previous sessions

## How to Use

1. Extract the user's search intent
2. Run the search command:
   ```bash
   cd ~/apps/claude-code/mneme && python -m mneme.cli search "user's query"
   ```
3. Present the formatted results
4. Offer to load full session context if needed

## Example

User: "What did we do with the LanceDB migration?"

Response includes:
- Matching session dates and summaries
- Relevant excerpts
- Session IDs for drill-down
```

---

## Testing Strategy

### Unit Tests
- `test_parser.py`: JSONL parsing, content extraction
- `test_embeddings.py`: Embedding generation
- `test_search.py`: Vector search, result formatting

### Integration Tests
- Index a sample session directory
- Search and verify results
- Test incremental updates

### Test Fixtures
- Sample JSONL files with various content types
- Edge cases: empty sessions, very long messages, tool-heavy sessions

---

## Performance Considerations

1. **Lazy Loading:** Embedding model loads on first use, not import
2. **Batched Embedding:** Process chunks in batches of 100
3. **Incremental Updates:** Skip unchanged files using mtime
4. **Result Caching:** Consider caching recent queries (optional)

---

## Future Enhancements

1. **MCP Server:** Expose as MCP tool for direct integration
2. **Session Summarization:** Auto-generate summaries for sessions without them
3. **Temporal Queries:** "What did we do last week?" with date filtering
4. **Project Scoping:** Search within specific project paths
