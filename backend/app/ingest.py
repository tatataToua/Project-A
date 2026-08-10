import argparse
from pathlib import Path

from sqlalchemy import delete

from app.chunking import chunk_markdown
from app.config import CONTENT_DIR
from app.db import SessionLocal
from app.embeddings import embed_texts
from app.models import EmbeddingChunk, Tenant, create_tables

CONTENT_FILES = ["bio.md", "projects.md", "resume.md"]


def tenant_content_dir(slug: str) -> Path:
    return CONTENT_DIR / "tenants" / slug


def get_or_create_tenant(session, slug: str) -> Tenant:
    tenant = session.query(Tenant).filter_by(slug=slug).one_or_none()
    if tenant is None:
        tenant = Tenant(slug=slug, name=slug, status="active")
        session.add(tenant)
        session.flush()
    return tenant


def ingest_tenant(slug: str) -> int:
    create_tables()
    content_dir = tenant_content_dir(slug)
    session = SessionLocal()
    try:
        tenant = get_or_create_tenant(session, slug)
        session.execute(delete(EmbeddingChunk).where(EmbeddingChunk.tenant_id == tenant.id))

        all_chunks = []
        for filename in CONTENT_FILES:
            file_path = content_dir / filename
            if not file_path.exists():
                continue
            text = file_path.read_text(encoding="utf-8")
            all_chunks.extend(chunk_markdown(text, filename))

        if not all_chunks:
            session.commit()
            return 0

        vectors = embed_texts([c.text for c in all_chunks])
        for chunk, vector in zip(all_chunks, vectors):
            session.add(
                EmbeddingChunk(
                    tenant_id=tenant.id,
                    source_file=chunk.source_file,
                    chunk_index=chunk.chunk_index,
                    chunk_text=chunk.text,
                    embedding=vector,
                )
            )
        session.commit()
        return len(all_chunks)
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk, embed, and store a tenant's content.")
    parser.add_argument("tenant_slug", nargs="?", default="toua")
    args = parser.parse_args()
    n = ingest_tenant(args.tenant_slug)
    print(f"Ingested {n} chunks for tenant '{args.tenant_slug}'")
