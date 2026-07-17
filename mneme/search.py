"""Search API for Mneme."""

from dataclasses import dataclass
from datetime import datetime
import re
from functools import lru_cache
from pathlib import Path
from typing import List

from .config import load_config
from .embeddings import EmbeddingModel
from .indexer import SessionVectorDB
from .rerank import CrossEncoderReranker


@dataclass
class SearchResult:
    """A single search result."""

    session_id: str
    session_file: str
    project_path: str
    timestamp: datetime
    chunk_type: str
    content: str
    similarity: float
    host: str = ""


@lru_cache(maxsize=4)
def _get_db(path: str, dimension: int) -> SessionVectorDB:
    """Reuse the DB handle across calls."""
    return SessionVectorDB(path, dimension=dimension)


@lru_cache(maxsize=4)
def _get_embedder(
    model_name: str,
    device: str | None,
    dimension: int,
    doc_prefix: str,
    query_prefix: str,
) -> EmbeddingModel:
    """Reuse the embedder across calls.

    EmbeddingModel loads its weights lazily and per-instance, so a fresh
    instance per search means re-loading ~560MB every time. A one-shot CLI
    never noticed; a long-lived HTTP server would do it on every request.
    Mirrors the lru_cache already used for the cross-encoder in rerank.py.
    """
    return EmbeddingModel(
        model_name=model_name,
        device=device,
        dimension=dimension,
        doc_prefix=doc_prefix,
        query_prefix=query_prefix,
    )


def _to_result(r: dict, similarity: float) -> SearchResult:
    """Map a raw DB row to a SearchResult.

    Every search mode builds results the same way and differs only in which
    score field it reads, so the mapping lives here once. Adding a column in
    four places is how one of them ends up forgotten.
    """
    return SearchResult(
        session_id=r.get("session_id", ""),
        session_file=r.get("session_file", ""),
        project_path=r.get("project_path", ""),
        timestamp=r.get("timestamp", datetime.now()),
        chunk_type=r.get("chunk_type", ""),
        content=r.get("content", ""),
        similarity=similarity,
        host=r.get("host", ""),
    )


@dataclass
class SessionMatch:
    """A session with matching chunks."""

    session_id: str
    session_file: str
    project_path: str
    timestamp: datetime
    best_similarity: float
    summary: str | None
    chunks: List[SearchResult]
    host: str = ""


def _host_filter(host: str | None) -> str | None:
    """Build a WHERE clause restricting results to one machine.

    Pushed into the query, NOT applied afterwards. Post-filtering a top-K cannot
    work for a minority host: the mini is ~3.6% of the corpus, so the top 80 chunks
    for any query contain almost no mini rows and a post-filter returns nothing.
    That reads as "you have no mini sessions about this" while 17,316 mini chunks
    sit in the table, which is a silent wrong answer rather than a thin one.

    The host is whitelisted to a conservative charset rather than escaped, because
    this string is interpolated into SQL and the set of real hostnames is small and
    known.
    """
    if not host:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", host):
        raise ValueError(f"invalid host filter: {host!r}")
    return f"host = '{host}'"


def search_sessions(
    query: str,
    limit: int = 10,
    config_path: str | None = None,
    mode: str = "vector",
    host: str | None = None,
) -> List[SearchResult]:
    """Search for sessions matching the query.

    Args:
        query: Natural language search query
        limit: Maximum number of chunk results
        config_path: Path to config file
        mode: Search mode - "vector" (semantic), "fts" (keyword/BM25),
            "hybrid" (both + RRF), or "rerank" (hybrid recall + local
            cross-encoder reranking)
        host: Restrict to one machine (mac | dgx | pi | mini)

    Returns:
        List of SearchResult objects ordered by relevance
    """
    fexpr = _host_filter(host)
    config = load_config(config_path)
    emb_cfg = config["embeddings"]
    dimension = emb_cfg.get("dimension", 768)
    db = _get_db(config["vectordb"]["path"], dimension)

    if mode == "fts":
        results = db.search_fts(query, limit=limit, filter_expr=fexpr)
        return [_to_result(r, r.get("_score", 0)) for r in results]

    # Vector or hybrid mode needs embeddings
    embedder = _get_embedder(
        emb_cfg["model"],
        emb_cfg.get("device"),
        dimension,
        emb_cfg.get("doc_prefix", "search_document: "),
        emb_cfg.get("query_prefix", "search_query: "),
    )
    query_vector = embedder.embed_query(query)

    if mode == "hybrid":
        results = db.search_hybrid(query, query_vector.tolist(), limit=limit, filter_expr=fexpr)
        return [_to_result(r, r.get("_relevance_score", 0)) for r in results]

    if mode == "rerank":
        # Stage 1: wide hybrid recall. Stage 2: local cross-encoder reorders.
        rcfg = config.get("rerank", {})
        model_name = rcfg.get("model", "BAAI/bge-reranker-v2-m3")
        n_candidates = max(rcfg.get("candidates", 50), limit)
        device = rcfg.get("device") or config["embeddings"].get("device")
        candidates = db.search_hybrid(
            query, query_vector.tolist(), limit=n_candidates, filter_expr=fexpr
        )
        reranker = CrossEncoderReranker(model_name, device=device)
        reranked = reranker.rerank(query, candidates, top_k=limit)
        return [_to_result(r, r.get("rerank_score", 0)) for r in reranked]

    # Default: vector search
    results = db.search(query_vector.tolist(), limit=limit, filter_expr=fexpr)
    return [_to_result(r, 1 - r.get("_distance", 1)) for r in results]


def search_and_group(
    query: str,
    limit: int = 20,
    max_sessions: int = 5,
    config_path: str | None = None,
    mode: str = "vector",
    host: str | None = None,
) -> List[SessionMatch]:
    """Search and group results by session.

    Args:
        query: Natural language search query
        limit: Maximum number of chunk results to search
        max_sessions: Maximum number of sessions to return
        config_path: Path to config file
        mode: Search mode - "vector", "fts", or "hybrid"

    Returns:
        List of SessionMatch objects with grouped chunks
    """
    results = search_sessions(query, limit=limit, config_path=config_path, mode=mode, host=host)

    if not results:
        return []

    # Group by session
    sessions: dict[str, SessionMatch] = {}

    for r in results:
        key = r.session_id or r.session_file

        if key not in sessions:
            sessions[key] = SessionMatch(
                session_id=r.session_id,
                session_file=r.session_file,
                project_path=r.project_path,
                timestamp=r.timestamp,
                best_similarity=r.similarity,
                summary=None,
                chunks=[],
                host=r.host,
            )

        match = sessions[key]
        match.chunks.append(r)

        # Track best similarity
        if r.similarity > match.best_similarity:
            match.best_similarity = r.similarity

        # Capture summary if found
        if r.chunk_type == "summary" and not match.summary:
            match.summary = r.content

    # Sort by best similarity and limit
    sorted_sessions = sorted(
        sessions.values(),
        key=lambda s: s.best_similarity,
        reverse=True,
    )[:max_sessions]

    return sorted_sessions


def format_results(results: List[SearchResult], max_content: int = 200) -> str:
    """Format search results for display.

    Args:
        results: List of SearchResult objects
        max_content: Maximum content length per result

    Returns:
        Formatted string for display
    """
    if not results:
        return "No matching sessions found."

    lines = ["**Search Results**", ""]

    for r in results:
        # Format timestamp
        if isinstance(r.timestamp, datetime):
            date_str = r.timestamp.strftime("%Y-%m-%d %H:%M")
        else:
            date_str = str(r.timestamp)[:16]

        # Truncate content
        content = r.content
        if len(content) > max_content:
            content = content[:max_content] + "..."

        # Format project path nicely (replace home directory with ~)
        project = r.project_path
        home = str(Path.home())
        if project.startswith(home):
            project = "~" + project[len(home):]

        score_str = f"{r.similarity:.2f}" if r.similarity <= 1 else f"{r.similarity:.1f}"
        lines.append(f"**{r.chunk_type}** ({score_str}) - {date_str}")
        lines.append(f"Project: {project}")
        lines.append(f"> {content}")
        lines.append("")

    return "\n".join(lines)


def format_sessions(sessions: List[SessionMatch], max_content: int = 300) -> str:
    """Format grouped session results for display.

    Args:
        sessions: List of SessionMatch objects
        max_content: Maximum content length per session

    Returns:
        Formatted string for display
    """
    if not sessions:
        return "No matching sessions found."

    lines = []

    for i, session in enumerate(sessions, 1):
        # Format timestamp
        if isinstance(session.timestamp, datetime):
            date_str = session.timestamp.strftime("%Y-%m-%d")
        else:
            date_str = str(session.timestamp)[:10]

        # Format project path (replace home directory with ~)
        project = session.project_path
        home = str(Path.home())
        if project.startswith(home):
            project = "~" + project[len(home):]

        # Host matters in a fleet index: the same project path exists on more
        # than one machine, so "~/foo" alone does not say where this ran.
        where = f"{session.host}:{project}" if session.host else project
        lines.append(f"### {i}. {date_str} - {where}")
        lines.append(f"Similarity: {session.best_similarity:.2f}")

        # Show summary or best chunk
        if session.summary:
            summary = session.summary
            if len(summary) > max_content:
                summary = summary[:max_content] + "..."
            lines.append(f"> {summary}")
        elif session.chunks:
            # Use best chunk as summary
            best_chunk = max(session.chunks, key=lambda c: c.similarity)
            content = best_chunk.content
            if len(content) > max_content:
                content = content[:max_content] + "..."
            lines.append(f"> {content}")

        lines.append(f"Session: `{session.session_id[:8] if session.session_id else 'N/A'}...`")
        lines.append("")

    return "\n".join(lines)
