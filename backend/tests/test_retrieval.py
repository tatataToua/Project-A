"""Requires Postgres running: `docker compose up -d` from the repo root."""
from app import retrieval
from app.config import EMBEDDING_DIMENSIONS
from app.db import SessionLocal
from app.models import EmbeddingChunk, Tenant, create_tables
from app.retrieval import retrieve_chunks

TENANT_A_SLUG = "test-retrieval-tenant-a"
TENANT_B_SLUG = "test-retrieval-tenant-b"


def _vector(seed: int, dim: int = EMBEDDING_DIMENSIONS) -> list[float]:
    v = [0.0] * dim
    v[seed % dim] = 1.0
    return v


def _passthrough_rerank(query_text, candidates, k):
    """Stand-in for the real cross-encoder: preserves whatever order the fused
    hybrid results already arrived in, just truncating to k and dropping the
    id. Used so these tests exercise the SQL/RRF logic in isolation,
    deterministically, without depending on the real reranker."""
    return [(src, chunk_text) for _chunk_id, src, chunk_text in candidates[:k]]


def _cleanup():
    session = SessionLocal()
    try:
        for slug in (TENANT_A_SLUG, TENANT_B_SLUG):
            tenant = session.query(Tenant).filter_by(slug=slug).one_or_none()
            if tenant is not None:
                session.query(EmbeddingChunk).filter_by(tenant_id=tenant.id).delete()
                session.query(Tenant).filter_by(id=tenant.id).delete()
        session.commit()
    finally:
        session.close()


def _seed():
    create_tables()
    session = SessionLocal()
    try:
        tenant_a = Tenant(slug=TENANT_A_SLUG, name="A", status="active")
        tenant_b = Tenant(slug=TENANT_B_SLUG, name="B", status="active")
        session.add_all([tenant_a, tenant_b])
        session.flush()

        session.add_all(
            [
                EmbeddingChunk(
                    tenant_id=tenant_a.id, source_file="bio.md", chunk_index=0,
                    chunk_text="tenant A chunk 0", embedding=_vector(0),
                ),
                EmbeddingChunk(
                    tenant_id=tenant_a.id, source_file="bio.md", chunk_index=1,
                    chunk_text="tenant A chunk 1", embedding=_vector(1),
                ),
                EmbeddingChunk(
                    tenant_id=tenant_b.id, source_file="bio.md", chunk_index=0,
                    chunk_text="tenant B chunk 0", embedding=_vector(0),
                ),
            ]
        )
        session.commit()
        return tenant_a.id, tenant_b.id
    finally:
        session.close()


def test_retrieval_never_crosses_tenants(monkeypatch):
    monkeypatch.setattr(retrieval, "rerank_chunks", _passthrough_rerank)
    _cleanup()
    tenant_a_id, tenant_b_id = _seed()
    try:
        results = retrieve_chunks(tenant_a_id, "chunk", _vector(0), k=5)
        assert len(results) == 2
        assert all("tenant A" in text for _src, text in results)
        assert not any("tenant B" in text for _src, text in results)
    finally:
        _cleanup()


def test_retrieval_orders_by_similarity_when_no_keyword_signal(monkeypatch):
    monkeypatch.setattr(retrieval, "rerank_chunks", _passthrough_rerank)
    _cleanup()
    tenant_a_id, _ = _seed()
    try:
        results = retrieve_chunks(tenant_a_id, "xyz", _vector(0), k=1)
        assert results == [("bio.md", "tenant A chunk 0")]
    finally:
        _cleanup()


def test_fulltext_signal_can_outrank_pure_vector_similarity(monkeypatch):
    monkeypatch.setattr(retrieval, "rerank_chunks", _passthrough_rerank)
    _cleanup()
    create_tables()
    session = SessionLocal()
    try:
        tenant = Tenant(slug=TENANT_A_SLUG, name="A", status="active")
        session.add(tenant)
        session.flush()
        session.add_all(
            [
                EmbeddingChunk(
                    tenant_id=tenant.id, source_file="bio.md", chunk_index=0,
                    chunk_text="unrelated filler text", embedding=_vector(0),
                ),
                EmbeddingChunk(
                    tenant_id=tenant.id, source_file="bio.md", chunk_index=1,
                    chunk_text="the tavern has a dedicated karaoke night",
                    embedding=_vector(500),
                ),
            ]
        )
        session.commit()
        tenant_id = tenant.id
    finally:
        session.close()

    try:
        results = retrieve_chunks(tenant_id, "karaoke night", _vector(0), k=1)
        assert results == [("bio.md", "the tavern has a dedicated karaoke night")]
    finally:
        _cleanup()


def test_reranker_is_actually_invoked_end_to_end():
    """No monkeypatch here -- proves the real cross-encoder is wired into
    retrieve_chunks. Downloads the model on first run (network + local cache)."""
    _cleanup()
    create_tables()
    session = SessionLocal()
    try:
        tenant = Tenant(slug=TENANT_A_SLUG, name="A", status="active")
        session.add(tenant)
        session.flush()
        session.add_all(
            [
                EmbeddingChunk(
                    tenant_id=tenant.id, source_file="about.md", chunk_index=0,
                    chunk_text="We are open from 5pm to 10pm Tuesday through Sunday.",
                    embedding=_vector(0),
                ),
                EmbeddingChunk(
                    tenant_id=tenant.id, source_file="about.md", chunk_index=1,
                    chunk_text="Our head chef previously worked in Lyon, France.",
                    embedding=_vector(1),
                ),
            ]
        )
        session.commit()
        tenant_id = tenant.id
    finally:
        session.close()

    try:
        results = retrieve_chunks(tenant_id, "What are your hours?", _vector(0), k=1)
        assert results == [("about.md", "We are open from 5pm to 10pm Tuesday through Sunday.")]
    finally:
        _cleanup()
