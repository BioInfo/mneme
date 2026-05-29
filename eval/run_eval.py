#!/usr/bin/env python
"""Evaluation harness for Mneme.

Measures retrieval quality (recall@k + MRR) of each search mode against a
labeled query set, so changes to the embedder, reranker, or chunking are
judged by numbers rather than vibes.

A query set is a JSONL file where each line maps a natural-language query to
the session that *should* be retrieved:

    {"query": "what did we do with the vector db migration", "session_id": "abcd1234-..."}

The query set lives at eval/queries.jsonl (gitignored — it references your own
session IDs). Generate one from your own index with make_queryset.py, or copy
queries.example.jsonl as a starting point.

Usage:
    ./venv/bin/python eval/run_eval.py
    ./venv/bin/python eval/run_eval.py --modes vector hybrid --k 10
    ./venv/bin/python eval/run_eval.py --save        # record a baseline
    ./venv/bin/python eval/run_eval.py --json         # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the package importable when run as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mneme.search import search_and_group  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_QUERYSET = EVAL_DIR / "queries.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
MODES = ("vector", "fts", "hybrid", "rerank")
RECALL_KS = (1, 5, 10)


def load_queryset(path: Path) -> list[dict]:
    """Load a JSONL query set. Each row needs 'query' and 'session_id'."""
    if not path.exists():
        sys.exit(
            f"Query set not found: {path}\n"
            "Create one with: ./venv/bin/python eval/make_queryset.py\n"
            "or copy eval/queries.example.jsonl to eval/queries.jsonl and edit it."
        )
    rows = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            sys.exit(f"{path}:{n}: invalid JSON ({e})")
        if "query" not in row or "session_id" not in row:
            sys.exit(f"{path}:{n}: each row needs 'query' and 'session_id'")
        rows.append(row)
    if not rows:
        sys.exit(f"{path}: no usable rows.")
    return rows


def rank_of_target(
    query: str, target: str, mode: str, k: int, config_path: str | None = None
) -> int | None:
    """Return the 1-based rank of the target session in the top-k, or None."""
    sessions = search_and_group(
        query, limit=max(k * 4, 20), max_sessions=k, mode=mode, config_path=config_path
    )
    for rank, s in enumerate(sessions, 1):
        if s.session_id == target:
            return rank
    return None


def score_mode(
    queryset: list[dict], mode: str, k: int, config_path: str | None = None
) -> dict:
    """Run every query in one mode and aggregate recall@k + MRR."""
    n = len(queryset)
    hits_at = {kk: 0 for kk in RECALL_KS if kk <= k}
    reciprocal_sum = 0.0
    misses: list[str] = []

    for row in queryset:
        rank = rank_of_target(row["query"], row["session_id"], mode, k, config_path)
        if rank is None:
            misses.append(row["query"])
            continue
        reciprocal_sum += 1.0 / rank
        for kk in hits_at:
            if rank <= kk:
                hits_at[kk] += 1

    return {
        "mode": mode,
        "n": n,
        "recall": {f"@{kk}": round(hits_at[kk] / n, 3) for kk in hits_at},
        "mrr": round(reciprocal_sum / n, 3),
        "misses": misses,
    }


def print_table(results: list[dict], k: int) -> None:
    ks = [kk for kk in RECALL_KS if kk <= k]
    header = f"{'mode':<8}" + "".join(f"R@{kk:<6}" for kk in ks) + f"{'MRR':<8}{'n':<5}"
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r['mode']:<8}"
        row += "".join(f"{r['recall'][f'@{kk}']:<8}" for kk in ks)
        row += f"{r['mrr']:<8}{r['n']:<5}"
        print(row)
    best = max(results, key=lambda r: r["mrr"])
    print(f"\nbest by MRR: {best['mode']} ({best['mrr']})")
    for r in results:
        if r["misses"]:
            print(f"\n{r['mode']} missed {len(r['misses'])}/{r['n']}:")
            for q in r["misses"]:
                print(f"  - {q}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Mneme search quality.")
    ap.add_argument("--queryset", type=Path, default=DEFAULT_QUERYSET)
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--k", type=int, default=10, help="Retrieval depth (default 10).")
    ap.add_argument("--save", action="store_true", help="Write results to eval/results/.")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    ap.add_argument(
        "--config", type=str, default=None,
        help="Path to an alternate config (e.g. an A/B index). Default: config.yaml.",
    )
    ap.add_argument(
        "--label", type=str, default=None,
        help="Tag recorded in saved results (e.g. 'bge-m3').",
    )
    args = ap.parse_args()

    queryset = load_queryset(args.queryset)
    results = [score_mode(queryset, mode, args.k, args.config) for mode in args.modes]

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "queryset": str(args.queryset.name),
        "k": args.k,
        "config": args.config or "config.yaml",
        "label": args.label,
        "results": results,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_table(results, args.k)

    if args.save:
        RESULTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        tag = f"-{args.label}" if args.label else ""
        out = RESULTS_DIR / f"{stamp}{tag}.json"
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nsaved results -> {out}")


if __name__ == "__main__":
    main()
