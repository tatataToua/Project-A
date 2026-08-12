"""Cross-encoder reranking: rescores retrieval candidates against the query.

The model *weights* are loaded lazily (not at import time) -- `_get_model`
only constructs the `CrossEncoder` on first use. The `sentence_transformers`
import itself (which pulls in torch) is also deferred into `_get_model` so
that importing this module (e.g. via the main -> tracing -> workflow ->
retrieval -> reranking chain on every app/test startup) doesn't pay that cost
for code paths that never call `rerank_chunks`.
"""
from typing import TYPE_CHECKING

from app.config import RERANK_MODEL

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

_model: "CrossEncoder | None" = None


def _get_model() -> "CrossEncoder":
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

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
