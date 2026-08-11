from sqlalchemy import select

from app.config import RETRIEVAL_TOP_K
from app.db import SessionLocal
from app.models import EmbeddingChunk


def retrieve_chunks(tenant_id: int, query_embedding: list[float], k: int = RETRIEVAL_TOP_K) -> list[str]:
    session = SessionLocal()
    try:
        stmt = (
            select(EmbeddingChunk.chunk_text)
            .where(EmbeddingChunk.tenant_id == tenant_id)
            .order_by(EmbeddingChunk.embedding.cosine_distance(query_embedding))
            .limit(k)
        )
        return list(session.scalars(stmt))
    finally:
        session.close()
