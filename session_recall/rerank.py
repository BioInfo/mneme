"""Local cross-encoder reranking for Session Recall.

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


class CrossEncoderReranker:
    """Reorders retrieved chunks by joint (query, content) relevance."""

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device

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
        if not candidates:
            return []
        model = _load(self.model_name, self.device)
        pairs = [(query, c.get("content", "")) for c in candidates]
        scores = model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]
