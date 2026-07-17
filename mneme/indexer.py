"""LanceDB indexer for Mneme."""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
import hashlib
import json
import os

import lancedb
import pyarrow as pa
from tqdm import tqdm

from .config import load_config
from .embeddings import EmbeddingModel
from .parser import SessionChunk, discover_session_files, parse_session_file


class SourceError(RuntimeError):
    """A required config source is missing or empty.

    Raised so the indexer exits non-zero instead of quietly indexing a subset
    of the fleet and reporting success.
    """


class SessionVectorDB:
    """Vector database for session chunks using LanceDB."""

    TABLE_NAME = "session_chunks"

    def __init__(self, db_path: str, dimension: int = 768):
        """Initialize the vector database.

        Args:
            db_path: Path to the LanceDB database directory
            dimension: Embedding vector dimensionality (must match the embedder).
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        self.dimension = dimension
        self._table = None

    def _get_schema(self) -> pa.Schema:
        """Get the PyArrow schema for the table."""
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("session_id", pa.string()),
            pa.field("session_file", pa.string()),
            pa.field("project_path", pa.string()),
            pa.field("timestamp", pa.timestamp("us", tz="UTC")),
            pa.field("chunk_type", pa.string()),
            pa.field("content", pa.string()),
            pa.field("host", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.dimension)),
        ])

    def _ensure_table(self):
        """Create table if it doesn't exist."""
        if self.TABLE_NAME not in self.db.table_names():
            self.db.create_table(self.TABLE_NAME, schema=self._get_schema())

    @property
    def table(self):
        """Get or create the table."""
        if self._table is None:
            self._ensure_table()
            self._table = self.db.open_table(self.TABLE_NAME)
        return self._table

    def add_chunks(self, chunks: list[dict[str, Any]]):
        """Add chunks to the database.

        Args:
            chunks: List of chunk dictionaries with 'vector' field
        """
        if not chunks:
            return
        self.table.add(chunks)

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filter_expr: str | None = None,
    ) -> list[dict]:
        """Vector search for similar chunks.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            filter_expr: Optional SQL filter expression

        Returns:
            List of result dictionaries with '_distance' field (cosine distance 0-2)
        """
        search = self.table.search(query_vector).distance_type("cosine").limit(limit)
        if filter_expr:
            search = search.where(filter_expr)
        return search.to_list()

    def search_fts(
        self,
        query: str,
        limit: int = 10,
        filter_expr: str | None = None,
    ) -> list[dict]:
        """Full-text (BM25) keyword search.

        Args:
            query: Text query for keyword matching
            limit: Maximum number of results
            filter_expr: Optional SQL filter expression

        Returns:
            List of result dictionaries with '_score' field
        """
        search = self.table.search(query, query_type="fts").limit(limit)
        if filter_expr:
            search = search.where(filter_expr)
        return search.to_list()

    def search_hybrid(
        self,
        query: str,
        query_vector: list[float],
        limit: int = 10,
        filter_expr: str | None = None,
    ) -> list[dict]:
        """Hybrid search combining vector + FTS with RRF reranking.

        Args:
            query: Text query for FTS component
            query_vector: Query embedding for vector component
            limit: Maximum number of results
            filter_expr: Optional SQL filter expression

        Returns:
            List of result dictionaries with '_relevance_score' field
        """
        from lancedb.rerankers import RRFReranker

        reranker = RRFReranker()
        # LanceDB hybrid: set the vector and text legs explicitly via the
        # builder. Passing the vector positionally to search() while also
        # calling .text() raises ("provide a string query OR set vector()
        # and text(), but not both").
        search = (
            self.table.search(query_type="hybrid")
            .vector(query_vector)
            .text(query)
            .limit(limit)
            .rerank(reranker=reranker)
        )
        if filter_expr:
            search = search.where(filter_expr)
        return search.to_list()

    def create_fts_index(self):
        """Create or replace the full-text search index on content."""
        self.table.create_fts_index("content", replace=True)
        print("FTS index created on 'content' column.")

    def delete_by_session_file(self, session_file: str):
        """Delete all chunks from a specific session file.

        Args:
            session_file: Path to the session file
        """
        self.table.delete(f"session_file = '{session_file}'")

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        try:
            count = self.table.count_rows()
        except Exception:
            count = 0

        return {
            "total_chunks": count,
            "db_path": str(self.db_path),
        }

    def clear(self):
        """Clear all data from the table (for full reindex)."""
        if self.TABLE_NAME in self.db.table_names():
            self.db.drop_table(self.TABLE_NAME)
            self._table = None


class IndexState:
    """Track indexing state for incremental updates."""

    def __init__(self, state_path: str):
        """Initialize index state tracker.

        Args:
            state_path: Path to the state JSON file
        """
        self.state_path = Path(state_path)
        self._state = self._load()

    def _load(self) -> dict:
        """Load state from file."""
        if self.state_path.exists():
            with open(self.state_path) as f:
                return json.load(f)
        return {
            "version": 1,
            "last_full_index": None,
            "last_incremental": None,
            "indexed_files": {},
            "stats": {
                "total_files": 0,
                "total_chunks": 0,
            },
        }

    def save(self):
        """Save state to file."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self._state, f, indent=2, default=str)

    def get_file_mtime(self, file_path: str) -> float | None:
        """Get the stored mtime for a file."""
        return self._state["indexed_files"].get(file_path, {}).get("mtime")

    def set_file_indexed(
        self,
        file_path: str,
        mtime: float,
        chunk_count: int,
    ):
        """Mark a file as indexed."""
        self._state["indexed_files"][file_path] = {
            "mtime": mtime,
            "chunks": chunk_count,
            "indexed_at": datetime.now().isoformat(),
        }

    def remove_file(self, file_path: str):
        """Remove a file from the index state."""
        self._state["indexed_files"].pop(file_path, None)

    def needs_reindex(self, file_path: str) -> bool:
        """Check if a file needs to be reindexed.

        Returns True if file is new or modified since last index.
        """
        path = Path(file_path)
        if not path.exists():
            return False

        current_mtime = path.stat().st_mtime
        stored_mtime = self.get_file_mtime(file_path)

        if stored_mtime is None:
            return True

        return current_mtime > stored_mtime

    def update_stats(self, total_files: int, total_chunks: int):
        """Update aggregate statistics."""
        self._state["stats"]["total_files"] = total_files
        self._state["stats"]["total_chunks"] = total_chunks

    def mark_incremental_complete(self):
        """Mark an incremental index as complete."""
        self._state["last_incremental"] = datetime.now().isoformat()

    def mark_full_index_complete(self):
        """Mark a full index as complete."""
        self._state["last_full_index"] = datetime.now().isoformat()
        self._state["last_incremental"] = datetime.now().isoformat()

    @property
    def stats(self) -> dict:
        """Get current statistics."""
        return self._state["stats"]


def generate_chunk_id(chunk: SessionChunk) -> str:
    """Generate a unique ID for a chunk."""
    # Use content hash + session + type for uniqueness
    content_hash = hashlib.md5(
        f"{chunk.session_file}:{chunk.chunk_type}:{chunk.content[:100]}".encode()
    ).hexdigest()[:12]
    return f"{chunk.session_id[:8]}_{chunk.chunk_type}_{content_hash}"


def run_indexer(
    config_path: str | None = None,
    full: bool = False,
    source_path: str | None = None,
):
    """Run the indexer to process session files.

    Args:
        config_path: Path to config file (default: config.yaml)
        full: Force full reindex
        source_path: Specific source path to index
    """
    config = load_config(config_path)
    emb_cfg = config["embeddings"]
    dimension = emb_cfg.get("dimension", 768)

    # Initialize components
    db = SessionVectorDB(config["vectordb"]["path"], dimension=dimension)
    state = IndexState(
        str(Path(config["vectordb"]["path"]).parent / "index_state.json")
    )
    embedder = EmbeddingModel(
        model_name=emb_cfg["model"],
        device=emb_cfg.get("device"),
        dimension=dimension,
        doc_prefix=emb_cfg.get("doc_prefix", "search_document: "),
        query_prefix=emb_cfg.get("query_prefix", "search_query: "),
    )

    indexing_config = config.get("indexing", {})
    batch_size = indexing_config.get("batch_size", 100)
    min_length = indexing_config.get("min_content_length", 20)
    max_length = indexing_config.get("max_content_length", 2000)

    # (path, host) pairs. The host has to survive all the way to the parse call:
    # drop it here and every row lands with an empty host, which is the whole
    # reason a fleet index can't tell one machine's sessions from another's.
    files_to_process: list[tuple[Path, str]] = []
    sources = config.get("sources", [])

    if source_path:
        # Index specific path
        sources = [{"path": source_path, "name": "specified", "optional": False}]

    # A missing REQUIRED source used to print "Warning:" and continue, exactly
    # like an optional one. Nothing raised, exit stayed 0. That is how the mac
    # indexed zero DGX sessions for months and never once said so. A required
    # source that isn't there, or that yields nothing, is now a hard failure.
    failures: list[str] = []

    for source in sources:
        path = source["path"]
        host = source.get("name", "")
        optional = source.get("optional", False)

        if not Path(path).exists():
            if optional:
                print(f"Skipping optional source (not found): {path}")
            else:
                failures.append(f"required source not found: {path} (name={host!r})")
            continue

        found = 0
        for jsonl_file in discover_session_files(path):
            found += 1
            file_str = str(jsonl_file)
            if full or state.needs_reindex(file_str):
                files_to_process.append((jsonl_file, host))

        print(f"Source {host or '(unnamed)'}: {found} session files under {path}")
        if found == 0 and not optional:
            failures.append(
                f"required source yielded 0 session files: {path} (name={host!r})"
            )

    if failures:
        for f in failures:
            print(f"ERROR: {f}")
        raise SourceError(
            f"{len(failures)} required source(s) unusable; refusing to index a "
            "partial fleet. Fix the source(s) above, or mark them optional if "
            "their absence is genuinely acceptable."
        )

    if not files_to_process:
        print("No files need indexing.")
        return

    # For full reindex, clear existing data first
    if full:
        print("Clearing existing index for full reindex...")
        db.clear()
        state._state["indexed_files"] = {}

    print(f"Indexing {len(files_to_process)} files...")

    # Process files
    total_chunks = 0
    chunks_batch = []

    for file_path, host in tqdm(files_to_process, desc="Processing files"):
        file_str = str(file_path)
        file_chunks = []

        try:
            # Parse the file
            for chunk in parse_session_file(
                file_str,
                min_content_length=min_length,
                max_content_length=max_length,
                host=host,
            ):
                file_chunks.append(chunk)

        except Exception as e:
            print(f"Error parsing {file_path.name}: {e}")
            continue

        if not file_chunks:
            # Mark as indexed even if empty.
            # stat() can raise here: parse_session_file guards `if not path.exists()`
            # and returns nothing, so a file that vanished between the listing and
            # its turn lands in exactly this branch and then gets stat'd unguarded.
            # Sessions are live files on machines still being used, and a long index
            # is a wide window, so this is normal, not exceptional. It killed a
            # 52,611-file run at 40% (2026-07-17).
            try:
                state.set_file_indexed(file_str, file_path.stat().st_mtime, 0)
            except OSError:
                pass
            continue

        # For incremental updates, delete existing chunks from this file first
        if not full:
            try:
                db.delete_by_session_file(file_str)
            except Exception:
                pass  # Table might not exist yet or file not indexed before

        chunks_batch.extend(file_chunks)

        # Process batch when full
        if len(chunks_batch) >= batch_size:
            _index_batch(chunks_batch, embedder, db)
            total_chunks += len(chunks_batch)
            chunks_batch = []
            # Save state after each batch to survive OOM/crashes
            state.save()

        # Update state. Same guard as the empty-file branch above: the file can
        # disappear between being parsed and being recorded. Skipping the state
        # entry just means it is re-examined next run, which is correct and cheap;
        # crashing the whole index because one live session rotated a transcript is
        # not.
        try:
            state.set_file_indexed(file_str, file_path.stat().st_mtime, len(file_chunks))
        except OSError:
            pass

    # Process remaining chunks
    if chunks_batch:
        _index_batch(chunks_batch, embedder, db)
        total_chunks += len(chunks_batch)

    # Update state
    stats = db.get_stats()
    state.update_stats(len(state._state["indexed_files"]), stats["total_chunks"])

    if full:
        state.mark_full_index_complete()
    else:
        state.mark_incremental_complete()

    state.save()

    print(f"Indexed {total_chunks} chunks from {len(files_to_process)} files.")
    print(f"Total chunks in database: {stats['total_chunks']}")

    # Rebuild FTS index after indexing
    print("Building FTS index...")
    db.create_fts_index()


def _index_batch(
    chunks: list[SessionChunk],
    embedder: EmbeddingModel,
    db: SessionVectorDB,
):
    """Index a batch of chunks."""
    # Extract content for embedding
    texts = [c.content for c in chunks]

    # Generate embeddings
    vectors = embedder.embed(texts)

    # Prepare records for database
    records = []
    for chunk, vector in zip(chunks, vectors):
        # Convert timestamp to UTC
        ts = chunk.timestamp
        if ts.tzinfo is None:
            from datetime import timezone
            ts = ts.replace(tzinfo=timezone.utc)

        records.append({
            "id": generate_chunk_id(chunk),
            "session_id": chunk.session_id,
            "session_file": chunk.session_file,
            "project_path": chunk.project_path,
            "timestamp": ts,
            "chunk_type": chunk.chunk_type,
            "content": chunk.content,
            "host": chunk.host,
            "vector": vector.tolist() if hasattr(vector, "tolist") else list(vector),
        })

    # Add to database
    db.add_chunks(records)
