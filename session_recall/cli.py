"""Command-line interface for Session Recall."""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Session Recall - Search your Claude Code session history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  session-recall index                    # Incremental index
  session-recall index --full             # Full reindex
  session-recall search "LanceDB work"    # Search sessions
  session-recall stats                    # Show statistics
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index session files")
    index_parser.add_argument(
        "--full",
        action="store_true",
        help="Force full reindex (default: incremental)",
    )
    index_parser.add_argument(
        "--path",
        type=str,
        help="Specific source path to index",
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search sessions")
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query (natural language)",
    )
    search_parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=20,
        help="Maximum chunk results to search (default: 20)",
    )
    search_parser.add_argument(
        "--sessions",
        "-s",
        type=int,
        default=5,
        help="Maximum sessions to show (default: 5)",
    )
    search_parser.add_argument(
        "--raw",
        action="store_true",
        help="Show raw chunk results instead of grouped sessions",
    )
    search_parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["vector", "fts", "hybrid", "rerank"],
        default="vector",
        help="Search mode: vector (semantic), fts (keyword/BM25), hybrid (both + RRF), rerank (hybrid + local cross-encoder). Default: vector",
    )

    # Stats command
    subparsers.add_parser("stats", help="Show index statistics")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "stats":
        cmd_stats(args)


def cmd_index(args):
    """Run the indexer."""
    from .indexer import run_indexer

    print("Starting indexer...")
    run_indexer(
        config_path=args.config,
        full=args.full,
        source_path=args.path,
    )


def cmd_search(args):
    """Search sessions."""
    from .search import (
        format_results,
        format_sessions,
        search_and_group,
        search_sessions,
    )

    if args.raw:
        results = search_sessions(
            args.query,
            limit=args.limit,
            config_path=args.config,
            mode=args.mode,
        )
        print(format_results(results))
    else:
        sessions = search_and_group(
            args.query,
            limit=args.limit,
            max_sessions=args.sessions,
            config_path=args.config,
            mode=args.mode,
        )
        print(format_sessions(sessions))


def cmd_stats(args):
    """Show index statistics."""
    from .config import load_config
    from .indexer import IndexState, SessionVectorDB

    config = load_config(args.config)
    db_path = config["vectordb"]["path"]

    db = SessionVectorDB(db_path)
    stats = db.get_stats()

    state_path = str(Path(db_path).parent / "index_state.json")
    state = IndexState(state_path)

    print("Session Recall Statistics")
    print("=" * 40)
    print(f"Database path: {stats['db_path']}")
    print(f"Total chunks:  {stats['total_chunks']}")
    print(f"Indexed files: {state.stats.get('total_files', 'N/A')}")
    print(f"Last full index: {state._state.get('last_full_index', 'Never')}")
    print(f"Last incremental: {state._state.get('last_incremental', 'Never')}")


if __name__ == "__main__":
    main()
