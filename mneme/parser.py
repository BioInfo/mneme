"""JSONL parser for Claude Code session files."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator
import json
import re


@dataclass
class SessionChunk:
    """A chunk of content from a session file."""

    session_id: str
    session_file: str
    project_path: str
    timestamp: datetime
    chunk_type: str  # "summary" | "user" | "assistant"
    content: str
    message_id: str | None = None


def extract_project_from_path(file_path: str) -> str:
    """Extract project path from session file path.

    Session files are stored at:
    ~/.claude/projects/{encoded-project-path}/{session-id}.jsonl

    The encoded path uses '-' for '/' so:
    -Users-you-projects-myapp -> /Users/you/projects/myapp
    """
    path = Path(file_path)
    parent_name = path.parent.name

    # Convert encoded path back to real path
    if parent_name.startswith("-"):
        # Replace - with / and handle the leading -
        project = "/" + parent_name[1:].replace("-", "/")
        return project

    return parent_name


def parse_timestamp(ts: str | None) -> datetime:
    """Parse ISO 8601 timestamp."""
    if not ts:
        return datetime.now()
    try:
        # Handle ISO format: 2025-12-19T12:07:37.862Z
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return datetime.now()


def extract_user_content(content) -> str:
    """Extract text from user message content.

    Content can be:
    - str: Simple text message
    - list: Array of content blocks (often tool_result)

    We skip tool_result arrays as they're too noisy.
    """
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        # Check if it's all tool results - skip those
        texts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif item.get("type") == "tool_result":
                    # Skip tool results entirely
                    continue
        return " ".join(texts).strip()
    return ""


def parse_session_file(
    file_path: str,
    min_content_length: int = 20,
    max_content_length: int = 2000,
) -> Iterator[SessionChunk]:
    """Parse a session JSONL file and yield indexable chunks.

    Args:
        file_path: Path to the JSONL session file
        min_content_length: Minimum content length to include
        max_content_length: Maximum content length (truncates longer)

    Yields:
        SessionChunk objects for each indexable piece of content
    """
    path = Path(file_path)
    if not path.exists():
        return

    project_path = extract_project_from_path(file_path)

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            record_type = record.get("type")

            # Session summaries (highest value)
            if record_type == "summary":
                summary = record.get("summary", "")
                if summary and len(summary) >= min_content_length:
                    yield SessionChunk(
                        session_id=record.get("leafUuid", ""),
                        session_file=file_path,
                        project_path=project_path,
                        timestamp=datetime.now(),
                        chunk_type="summary",
                        content=summary[:max_content_length],
                    )

            # User messages
            elif record_type == "user":
                msg = record.get("message", {})
                content = extract_user_content(msg.get("content", ""))

                if content and len(content) >= min_content_length:
                    yield SessionChunk(
                        session_id=record.get("sessionId", ""),
                        session_file=file_path,
                        project_path=project_path,
                        timestamp=parse_timestamp(record.get("timestamp")),
                        chunk_type="user",
                        content=content[:max_content_length],
                        message_id=record.get("uuid"),
                    )

            # Assistant text responses
            elif record_type == "assistant":
                msg = record.get("message", {})
                content_blocks = msg.get("content", [])

                if not isinstance(content_blocks, list):
                    continue

                for block in content_blocks:
                    if not isinstance(block, dict):
                        continue

                    # Only index text blocks, skip thinking and tool_use
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if text and len(text) >= min_content_length:
                            yield SessionChunk(
                                session_id=record.get("sessionId", ""),
                                session_file=file_path,
                                project_path=project_path,
                                timestamp=parse_timestamp(record.get("timestamp")),
                                chunk_type="assistant",
                                content=text[:max_content_length],
                                message_id=record.get("uuid"),
                            )


def discover_session_files(source_path: str) -> Iterator[Path]:
    """Discover all JSONL session files in a source directory.

    Args:
        source_path: Path to search (e.g., ~/.claude/projects)

    Yields:
        Path objects for each .jsonl file found
    """
    source = Path(source_path)
    if not source.exists():
        return

    # Find all .jsonl files recursively
    for jsonl_file in source.rglob("*.jsonl"):
        # Skip tmp files
        if jsonl_file.name.endswith(".tmp"):
            continue
        yield jsonl_file
