"""Local cross-encoder reranking for Mneme.

First-stage retrieval (vector / fts / hybrid) is recall-oriented: it casts a
wide net cheaply. A cross-encoder then reads each (query, chunk) pair jointly
and produces a sharper relevance score, reordering the candidates. This is the
single biggest quality lever and runs entirely locally — no API, no DGX.

Default model is BAAI/bge-reranker-v2-m3 (same family the vault search uses).
It downloads (~560MB) on first use and caches. For a lighter footprint, set
rerank.model to cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB, faster, weaker).
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=2)
def _load(model_name: str, device: str | None):
    """Load and cache a CrossEncoder. Cached so repeated searches don't reload."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=device)


def _predict(model, pairs: list[tuple[str, str]]) -> list[float]:
    """Score pairs shortest-first so each batch pads to a similar length.

    predict() pads every batch to its longest member. Chunks run 20 to 2,000
    chars, so arrival order wastes most of each batch on padding. Sorting changes
    no score (max diff 5e-7, identical order on two real queries) and cut CPU
    rerank of 50 candidates from 33-38s to 23-27s.
    """
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][1]))
    scores = model.predict([pairs[i] for i in order], batch_size=16)
    out = [0.0] * len(pairs)
    for i, s in zip(order, scores):
        out[i] = float(s)
    return out


class CrossEncoderReranker:
    """Reorders retrieved chunks by joint (query, content) relevance."""

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device

    def score(self, query: str, candidates: list[dict]) -> list[dict]:
        """Annotate every candidate with 'rerank_score'; no sort, no cut.

        On CUDA this is ~2s for 50 candidates against ~25s on the DGX's CPU.
        A CUDA failure (init, OOM while another tenant holds the GPU) falls back
        to the CPU copy of the same weights, so a busy GPU costs latency, never
        an error. The fallback is printed so it is visible in the unit log.
        """
        if not candidates:
            return []
        pairs = [(query, c.get("content", "")) for c in candidates]
        if self.device and self.device.startswith("cuda"):
            try:
                scores = _predict(_load(self.model_name, self.device), pairs)
                import torch

                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001 - any CUDA fault -> CPU
                print(f"[mneme] rerank on {self.device} failed, using cpu: {e}", flush=True)
                scores = _predict(_load(self.model_name, "cpu"), pairs)
        else:
            scores = _predict(_load(self.model_name, self.device), pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = s
        return candidates

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Score each candidate against the query and return the top_k.

        Args:
            query: The search query.
            candidates: Result dicts (each must have a 'content' field).
            top_k: How many to return after reranking.

        Returns:
            The top_k candidates, sorted by cross-encoder score (desc), each
            annotated with a 'rerank_score' field.
        """
        scored = self.score(query, candidates)
        ranked = sorted(scored, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]
