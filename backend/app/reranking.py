"""Cross-encoder reranking: rescores retrieval candidates against the query.

The model is loaded lazily (not at import time) because it pulls in torch --
code paths that never call `rerank_chunks` (e.g. workflow tests that monkeypatch
retrieval entirely) shouldn't pay that startup cost.
"""
from sentence_transformers import CrossEncoder

from app.config import RERANK_MODEL

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def rerank_chunks(query_text: str, candidates: list[tuple[int, str]], k: int) -> list[str]:
    """Rescore (query, chunk_text) pairs with a cross-encoder and return the
    top-k chunk texts, best first."""
    if not candidates:
        return []
    pairs = [(query_text, chunk_text) for _chunk_id, chunk_text in candidates]
    scores = _get_model().predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk_text for (_chunk_id, chunk_text), _score in ranked[:k]]
