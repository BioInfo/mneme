#!/usr/bin/env python
"""HTTP front end for mneme, the fleet-wide Claude Code session index.

Runs on the DGX as `mneme-api.service` and is the only way the other machines
reach the index: the mac, mini and Pi hold no engine and no database, they just
POST here. Shaped after ~/workspace/infrastructure/search/search_server.py
(dgx-search, port 6335), which is the same idea over Qdrant for the vault.

  POST /search   {query, limit, sessions, mode, host}
  GET  /health   503 unless the index can actually answer
  GET  /stats    row counts, per-host breakdown, index freshness
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from mneme.config import load_config
from mneme.search import _get_db, _get_embedder, search_and_group, search_sessions

app = FastAPI(title="Mneme Session Search", version="1.0")

CONFIG_PATH: str | None = None
_config: dict = {}


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(20, ge=1, le=200, description="chunks to retrieve")
    sessions: int = Field(5, ge=1, le=50, description="sessions to return")
    mode: str = Field("rerank", description="vector | fts | hybrid | rerank")
    host: str | None = Field(
        None, description="restrict to one machine: mac | dgx | pi | mini"
    )
    raw: bool = Field(False, description="flat chunk list instead of grouped")


def _index_state() -> dict:
    """Read index_state.json next to the vector db."""
    p = Path(_config["vectordb"]["path"]).parent / "index_state.json"
    try:
        with open(p) as f:
            s = json.load(f)
        return {
            "last_full_index": s.get("last_full_index"),
            "last_incremental": s.get("last_incremental"),
            "indexed_files": len(s.get("indexed_files", {})),
        }
    except Exception:
        return {}


EXPECTED_HOSTS = ("mac", "dgx", "pi", "mini")


def _host_counts() -> dict[str, int]:
    """Rows per host, via count_rows(filter=...).

    One cheap counting query per host: no pylance dependency (table.to_lance()
    needs it and it is not installed), and no pulling a 100k+ row column into
    memory just to tally it.

    Returns {} only when the host column genuinely does not exist, i.e. an index
    written before that column. Any other failure propagates: /health swallowing
    a real error here would report "no hosts", which is indistinguishable from a
    fleet index that has quietly lost three machines.
    """
    db = _get_db(
        _config["vectordb"]["path"], _config["embeddings"].get("dimension", 768)
    )
    if "host" not in [f.name for f in db.table.schema]:
        return {}
    out: dict[str, int] = {}
    for h in EXPECTED_HOSTS:
        n = db.table.count_rows(filter=f"host = '{h}'")
        if n:
            out[h] = int(n)
    return out


@app.get("/health")
async def health():
    """Fail when the index cannot answer.

    dgx-search's /health returns {"status":"ok"} unconditionally, which cannot
    distinguish a healthy service from one serving an empty table. Report ok
    only when there are rows to search, so a monitor watching this endpoint is
    watching something that can actually go red.
    """
    try:
        stats = _get_db(
            _config["vectordb"]["path"], _config["embeddings"].get("dimension", 768)
        ).get_stats()
        chunks = int(stats.get("total_chunks", 0))
        hosts = _host_counts()
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": f"index unreadable: {e}"},
        )

    if chunks == 0:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "detail": "index is empty", "chunks": 0},
        )
    return {"status": "ok", "chunks": chunks, "hosts": hosts, **_index_state()}


@app.get("/stats")
async def stats():
    db = _get_db(
        _config["vectordb"]["path"], _config["embeddings"].get("dimension", 768)
    )
    return {"db": db.get_stats(), "hosts": _host_counts(), **_index_state()}


@app.post("/search")
async def search(req: SearchRequest):
    t0 = time.perf_counter()

    # Host filtering is a post-filter, so over-fetch to still return `limit`
    # rows after the other machines are dropped. Rough but honest: with four
    # hosts a 4x over-fetch usually suffices, and under-filling is visible in
    # the returned count rather than silently wrong.
    fetch = req.limit * 4 if req.host else req.limit

    try:
        if req.raw:
            results = search_sessions(
                req.query, limit=fetch, config_path=CONFIG_PATH, mode=req.mode
            )
            if req.host:
                results = [r for r in results if r.host == req.host][: req.limit]
            payload = {"results": [asdict(r) for r in results], "count": len(results)}
        else:
            matches = search_and_group(
                req.query,
                limit=fetch,
                max_sessions=req.sessions if not req.host else req.sessions * 4,
                config_path=CONFIG_PATH,
                mode=req.mode,
            )
            if req.host:
                matches = [m for m in matches if m.host == req.host][: req.sessions]
            payload = {
                "sessions": [asdict(m) for m in matches],
                "count": len(matches),
            }
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"status": "error", "detail": str(e)}
        )

    payload["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    payload["mode"] = req.mode
    if req.host:
        payload["host_filter"] = req.host
    return payload


@app.on_event("startup")
async def startup():
    """Warm the model and DB once, not per request."""
    global _config
    _config = load_config(CONFIG_PATH)
    emb = _config["embeddings"]
    dim = emb.get("dimension", 768)
    _get_db(_config["vectordb"]["path"], dim)
    embedder = _get_embedder(
        emb["model"],
        emb.get("device"),
        dim,
        emb.get("doc_prefix", "search_document: "),
        emb.get("query_prefix", "search_query: "),
    )
    embedder.embed_query("warmup")
    print(f"[mneme-api] warm: embedder on {embedder._device}", flush=True)


def main():
    global CONFIG_PATH
    ap = argparse.ArgumentParser(description="Mneme session search HTTP API.")
    ap.add_argument("--port", type=int, default=6336)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--config", default=None, help="Path to config.yaml")
    args = ap.parse_args()

    CONFIG_PATH = args.config
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
