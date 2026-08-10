"""Requires Postgres running: `docker compose up -d` from the repo root."""
from app.config import EMBEDDING_DIMENSIONS
from app.db import SessionLocal
from app.models import EmbeddingChunk, Tenant, create_tables


def _vector(seed: int, dim: int = EMBEDDING_DIMENSIONS) -> list[float]:
    v = [0.0] * dim
    v[seed % dim] = 1.0
    return v


def test_create_tenant_and_embedding_chunk():
    create_tables()
    session = SessionLocal()
    try:
        session.query(EmbeddingChunk).delete()
        session.query(Tenant).filter(Tenant.slug == "test-models-tenant").delete()
        session.commit()

        tenant = Tenant(slug="test-models-tenant", name="Test Tenant", status="active")
        session.add(tenant)
        session.flush()

        chunk = EmbeddingChunk(
            tenant_id=tenant.id,
            source_file="bio.md",
            chunk_index=0,
            chunk_text="## Section\nSome text.",
            embedding=_vector(1),
        )
        session.add(chunk)
        session.commit()

        fetched = session.query(EmbeddingChunk).filter_by(tenant_id=tenant.id).one()
        assert fetched.source_file == "bio.md"
        assert fetched.chunk_text == "## Section\nSome text."
        assert len(fetched.embedding) == EMBEDDING_DIMENSIONS
    finally:
        session.query(EmbeddingChunk).delete()
        session.query(Tenant).filter(Tenant.slug == "test-models-tenant").delete()
        session.commit()
        session.close()
