#!/usr/bin/env python
"""Generate a starter query set from your own local index.

Reads the most recent indexed sessions and writes a template eval/queries.jsonl
with one row per session: the session_id is filled in, and a summary snippet is
included as a hint. You then replace each "TODO" with a natural-language query
you'd realistically type to find that session.

This keeps the *methodology* shareable while your actual query set (which
references your session IDs and topics) stays local and gitignored.

Usage:
    ./venv/bin/python eval/make_queryset.py            # 30 most recent sessions
    ./venv/bin/python eval/make_queryset.py --n 50
    ./venv/bin/python eval/make_queryset.py --out eval/queries.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from session_recall.config import load_config  # noqa: E402
from session_recall.indexer import SessionVectorDB  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent


def collect_sessions(db: SessionVectorDB, n: int) -> list[dict]:
    """Return the n most recent sessions, each with a representative snippet."""
    rows = db.table.to_arrow().to_pylist()
    by_session: dict[str, dict] = {}
    for r in rows:
        sid = r.get("session_id") or r.get("session_file", "")
        if not sid:
            continue
        cur = by_session.get(sid)
        # Prefer a summary chunk as the snippet; otherwise keep the first seen.
        if cur is None:
            by_session[sid] = r
        elif r.get("chunk_type") == "summary" and cur.get("chunk_type") != "summary":
            by_session[sid] = r
    ordered = sorted(
        by_session.values(),
        key=lambda r: r.get("timestamp") or 0,
        reverse=True,
    )
    return ordered[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a starter eval query set.")
    ap.add_argument("--n", type=int, default=30, help="Number of recent sessions.")
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "queries.jsonl")
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    if args.out.exists():
        resp = input(f"{args.out} exists. Overwrite? [y/N] ").strip().lower()
        if resp != "y":
            sys.exit("Aborted.")

    config = load_config(args.config)
    db = SessionVectorDB(config["vectordb"]["path"])
    sessions = collect_sessions(db, args.n)
    if not sessions:
        sys.exit("No sessions in the index. Run `index` first.")

    lines = []
    for s in sessions:
        snippet = (s.get("content") or "").replace("\n", " ").strip()[:140]
        lines.append(json.dumps({
            "query": "TODO: write a query you'd use to find this session",
            "session_id": s.get("session_id", ""),
            "_hint": snippet,
        }))

    args.out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} template rows -> {args.out}")
    print("Next: edit each 'query' field, then run ./venv/bin/python eval/run_eval.py")


if __name__ == "__main__":
    main()
