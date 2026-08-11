"""Requires Postgres running: `docker compose up -d` from the repo root."""
from app.config import EMBEDDING_DIMENSIONS
from app.db import SessionLocal
from app.models import EmbeddingChunk, Tenant, create_tables


TEST_SLUG = "test-models-tenant"


def _vector(seed: int, dim: int = EMBEDDING_DIMENSIONS) -> list[float]:
    v = [0.0] * dim
    v[seed % dim] = 1.0
    return v


def _cleanup():
    session = SessionLocal()
    try:
        tenant = session.query(Tenant).filter_by(slug=TEST_SLUG).one_or_none()
        if tenant is not None:
            session.query(EmbeddingChunk).filter_by(tenant_id=tenant.id).delete()
            session.query(Tenant).filter_by(id=tenant.id).delete()
            session.commit()
    finally:
        session.close()


def test_create_tenant_and_embedding_chunk():
    create_tables()
    _cleanup()
    session = SessionLocal()
    try:
        tenant = Tenant(slug=TEST_SLUG, name="Test Tenant", status="active")
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
        session.close()
        _cleanup()
