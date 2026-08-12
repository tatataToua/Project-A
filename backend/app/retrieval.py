import logging

from sqlalchemy import select, text

from app.config import RETRIEVAL_OVERFETCH_K, RETRIEVAL_TOP_K
from app.db import SessionLocal
from app.models import EmbeddingChunk
from app.reranking import rerank_chunks

logger = logging.getLogger("askme")

_RRF_CONSTANT = 60  # standard smoothing constant for reciprocal rank fusion


def _vector_search(session, tenant_id: int, query_embedding: list[float], k: int) -> list[tuple[int, str]]:
    stmt = (
        select(EmbeddingChunk.id, EmbeddingChunk.chunk_text)
        .where(EmbeddingChunk.tenant_id == tenant_id)
        .order_by(EmbeddingChunk.embedding.cosine_distance(query_embedding))
        .limit(k)
    )
    return [(row.id, row.chunk_text) for row in session.execute(stmt)]


def _fulltext_search(session, tenant_id: int, query_text: str, k: int) -> list[tuple[int, str]]:
    rows = session.execute(
        text(
            """
            SELECT id, chunk_text
            FROM embeddings
            WHERE tenant_id = :tenant_id
              AND chunk_tsv @@ plainto_tsquery('english', :query_text)
            ORDER BY ts_rank(chunk_tsv, plainto_tsquery('english', :query_text)) DESC
            LIMIT :k
            """
        ),
        {"tenant_id": tenant_id, "query_text": query_text, "k": k},
    )
    return [(row.id, row.chunk_text) for row in rows]


def _reciprocal_rank_fusion(ranked_lists: list[list[tuple[int, str]]], k: int) -> list[tuple[int, str]]:
    """Merge multiple ranked (id, text) lists into one, scoring each id by
    sum(1 / (_RRF_CONSTANT + rank)) across every list it appears in -- a chunk
    that ranks well in both vector and full-text search outranks one that only
    ranks well in a single list."""
    scores: dict[int, float] = {}
    texts: dict[int, str] = {}
    for ranked in ranked_lists:
        for rank, (chunk_id, chunk_text) in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (_RRF_CONSTANT + rank)
            texts[chunk_id] = chunk_text
    ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [(cid, texts[cid]) for cid in ordered_ids[:k]]


def retrieve_chunks(
    tenant_id: int,
    query_text: str,
    query_embedding: list[float],
    k: int = RETRIEVAL_TOP_K,
    overfetch_k: int = RETRIEVAL_OVERFETCH_K,
) -> list[str]:
    """Hybrid retrieval: fuse pgvector semantic search with Postgres full-text
    search (reciprocal rank fusion), then rerank the fused candidates with a
    cross-encoder down to the final top-k. Every branch is filtered by
    tenant_id -- the tenant isolation boundary."""
    overfetch_k = max(overfetch_k, k)
    session = SessionLocal()
    try:
        vector_hits = _vector_search(session, tenant_id, query_embedding, overfetch_k)
        fulltext_hits = _fulltext_search(session, tenant_id, query_text, overfetch_k)
        fused = _reciprocal_rank_fusion([vector_hits, fulltext_hits], k=overfetch_k)
        try:
            return rerank_chunks(query_text, fused, k)
        except Exception:
            logger.exception("Reranker failed; falling back to un-reranked hybrid results")
            return [chunk_text for _chunk_id, chunk_text in fused[:k]]
    finally:
        session.close()
