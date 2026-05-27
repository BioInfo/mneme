#!/usr/bin/env python
"""A/B a candidate embedder against the live index — in isolation.

Builds a SEPARATE index with the candidate embedder (the live db is never
touched), evaluates it with the same query set, and prints a side-by-side
delta against the most recent live baseline. Nothing is swapped: you read the
numbers and decide.

Example:
    ./venv/bin/python experiments/embedder_ab.py \
        --model BAAI/bge-m3 --dim 1024 --label bge-m3 \
        --doc-prefix "" --query-prefix ""

Artifacts land under data/ab-<label>/ (gitignored). Eval results are written to
eval/results/<stamp>-<label>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from session_recall.indexer import run_indexer, SessionVectorDB  # noqa: E402
import run_eval  # noqa: E402  (eval/run_eval.py)

RESULTS_DIR = ROOT / "eval" / "results"


def newest_baseline(exclude_label: str) -> dict | None:
    """Most recent saved results from the live config (no/other label)."""
    best = None
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            payload = json.loads(f.read_text())
        except Exception:
            continue
        if payload.get("label") == exclude_label:
            continue
        if payload.get("config", "config.yaml") != "config.yaml":
            continue
        best = payload  # sorted ascending -> last wins = newest
    return best


def print_delta(baseline: dict | None, candidate: dict) -> None:
    cand = {r["mode"]: r for r in candidate["results"]}
    print("\n=== candidate (" + (candidate.get("label") or "?") + ") ===")
    run_eval.print_table(candidate["results"], candidate["k"])
    if not baseline:
        print("\n(no live baseline found to compare against)")
        return
    base = {r["mode"]: r for r in baseline["results"]}
    print("\n=== delta vs live baseline (candidate - baseline) ===")
    print(f"{'mode':<8}{'dR@1':<8}{'dR@5':<8}{'dR@10':<8}{'dMRR':<8}")
    print("-" * 40)
    for mode in cand:
        if mode not in base:
            continue
        c, b = cand[mode], base[mode]
        d1 = c["recall"].get("@1", 0) - b["recall"].get("@1", 0)
        d5 = c["recall"].get("@5", 0) - b["recall"].get("@5", 0)
        d10 = c["recall"].get("@10", 0) - b["recall"].get("@10", 0)
        dm = c["mrr"] - b["mrr"]
        print(f"{mode:<8}{d1:<+8.3f}{d5:<+8.3f}{d10:<+8.3f}{dm:<+8.3f}")
    print("\nPositive = candidate better. Decide the swap from the rerank row.")


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B a candidate embedder in isolation.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--doc-prefix", default="")
    ap.add_argument("--query-prefix", default="")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--base-config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--modes", nargs="+", default=["vector", "fts", "hybrid", "rerank"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--mac-only", action="store_true",
        help="Drop optional sources (e.g. a remote mount) so the index is "
             "exactly the local corpus — guarantees an apples-to-apples compare.",
    )
    args = ap.parse_args()

    # 1. Derive an isolated config from the live one (same sources, new db + embedder).
    base = yaml.safe_load(Path(args.base_config).read_text())
    if args.mac_only:
        kept = [s for s in base.get("sources", []) if not s.get("optional")]
        dropped = [s.get("name", s.get("path")) for s in base.get("sources", []) if s.get("optional")]
        base["sources"] = kept
        print(f"[ab] mac-only: kept {[s.get('name') for s in kept]}, dropped {dropped}")
    work = ROOT / "data" / f"ab-{args.label}"
    work.mkdir(parents=True, exist_ok=True)
    base["vectordb"]["path"] = str(work / "lance")
    base["embeddings"] = {
        "model": args.model,
        "dimension": args.dim,
        "device": args.device,
        "doc_prefix": args.doc_prefix,
        "query_prefix": args.query_prefix,
    }
    iso_cfg = work / "config.yaml"
    iso_cfg.write_text(yaml.safe_dump(base))
    print(f"[ab] {datetime.now(timezone.utc).isoformat()}")
    print(f"[ab] model={args.model} dim={args.dim}")
    print(f"[ab] isolated db -> {base['vectordb']['path']}")

    # 2. Full index with the candidate embedder.
    print("[ab] indexing (full)...", flush=True)
    run_indexer(config_path=str(iso_cfg), full=True)

    # 3. FTS index (needed for fts/hybrid/rerank).
    SessionVectorDB(base["vectordb"]["path"], dimension=args.dim).create_fts_index()

    # 4. Evaluate against the isolated index.
    print("[ab] evaluating...", flush=True)
    queryset = run_eval.load_queryset(run_eval.DEFAULT_QUERYSET)
    results = [
        run_eval.score_mode(queryset, m, args.k, str(iso_cfg)) for m in args.modes
    ]
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "queryset": run_eval.DEFAULT_QUERYSET.name,
        "k": args.k,
        "config": str(iso_cfg),
        "label": args.label,
        "results": results,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"{stamp}-{args.label}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[ab] saved -> {out}")

    # 5. Compare to the live baseline.
    print_delta(newest_baseline(args.label), payload)


if __name__ == "__main__":
    main()
