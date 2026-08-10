"""Requires Postgres running: `docker compose up -d` from the repo root."""
from app import ingest
from app.db import SessionLocal
from app.models import EmbeddingChunk, Tenant

TEST_SLUG = "test-ingest-tenant"
TEST_SLUG_B = "test-ingest-tenant-b"


def _cleanup(slug=TEST_SLUG):
    session = SessionLocal()
    try:
        tenant = session.query(Tenant).filter_by(slug=slug).one_or_none()
        if tenant is not None:
            session.query(EmbeddingChunk).filter_by(tenant_id=tenant.id).delete()
            session.query(Tenant).filter_by(id=tenant.id).delete()
            session.commit()
    finally:
        session.close()


def test_ingest_tenant_creates_tenant_and_chunks(tmp_path, monkeypatch):
    _cleanup()
    monkeypatch.setattr(ingest, "CONTENT_DIR", tmp_path)
    monkeypatch.setattr(
        ingest, "embed_texts", lambda texts: [[0.0] * 768 for _ in texts]
    )

    content_dir = tmp_path / "tenants" / TEST_SLUG
    content_dir.mkdir(parents=True)
    (content_dir / "bio.md").write_text("## Section\nSome bio text.", encoding="utf-8")

    count = ingest.ingest_tenant(TEST_SLUG)

    assert count == 1
    session = SessionLocal()
    try:
        tenant = session.query(Tenant).filter_by(slug=TEST_SLUG).one()
        chunks = session.query(EmbeddingChunk).filter_by(tenant_id=tenant.id).all()
        assert len(chunks) == 1
        assert chunks[0].source_file == "bio.md"
        assert "Some bio text." in chunks[0].chunk_text
    finally:
        session.close()
        _cleanup()


def test_ingest_tenant_replaces_existing_chunks(tmp_path, monkeypatch):
    _cleanup()
    monkeypatch.setattr(ingest, "CONTENT_DIR", tmp_path)
    monkeypatch.setattr(
        ingest, "embed_texts", lambda texts: [[0.0] * 768 for _ in texts]
    )

    content_dir = tmp_path / "tenants" / TEST_SLUG
    content_dir.mkdir(parents=True)
    bio_path = content_dir / "bio.md"

    bio_path.write_text("## Old\nOld text.", encoding="utf-8")
    ingest.ingest_tenant(TEST_SLUG)

    bio_path.write_text("## New\nNew text.", encoding="utf-8")
    count = ingest.ingest_tenant(TEST_SLUG)

    assert count == 1
    session = SessionLocal()
    try:
        tenant = session.query(Tenant).filter_by(slug=TEST_SLUG).one()
        chunks = session.query(EmbeddingChunk).filter_by(tenant_id=tenant.id).all()
        assert len(chunks) == 1
        assert "New text." in chunks[0].chunk_text
    finally:
        session.close()
        _cleanup()


def test_ingest_tenant_does_not_touch_other_tenants_chunks(tmp_path, monkeypatch):
    _cleanup(TEST_SLUG)
    _cleanup(TEST_SLUG_B)
    monkeypatch.setattr(ingest, "CONTENT_DIR", tmp_path)
    monkeypatch.setattr(
        ingest, "embed_texts", lambda texts: [[0.0] * 768 for _ in texts]
    )

    try:
        content_dir_a = tmp_path / "tenants" / TEST_SLUG
        content_dir_a.mkdir(parents=True)
        (content_dir_a / "bio.md").write_text(
            "## Section A\nTenant A text.", encoding="utf-8"
        )

        content_dir_b = tmp_path / "tenants" / TEST_SLUG_B
        content_dir_b.mkdir(parents=True)
        (content_dir_b / "bio.md").write_text(
            "## Section B\nTenant B text.", encoding="utf-8"
        )

        ingest.ingest_tenant(TEST_SLUG)
        # Re-ingesting tenant B should not disturb tenant A's already-stored rows.
        ingest.ingest_tenant(TEST_SLUG_B)
        ingest.ingest_tenant(TEST_SLUG_B)

        session = SessionLocal()
        try:
            tenant_a = session.query(Tenant).filter_by(slug=TEST_SLUG).one()
            tenant_b = session.query(Tenant).filter_by(slug=TEST_SLUG_B).one()

            chunks_a = session.query(EmbeddingChunk).filter_by(tenant_id=tenant_a.id).all()
            chunks_b = session.query(EmbeddingChunk).filter_by(tenant_id=tenant_b.id).all()

            assert len(chunks_a) == 1
            assert "Tenant A text." in chunks_a[0].chunk_text
            assert chunks_a[0].tenant_id == tenant_a.id

            assert len(chunks_b) == 1
            assert "Tenant B text." in chunks_b[0].chunk_text
            assert chunks_b[0].tenant_id == tenant_b.id

            assert tenant_a.id != tenant_b.id
        finally:
            session.close()
    finally:
        _cleanup(TEST_SLUG)
        _cleanup(TEST_SLUG_B)
