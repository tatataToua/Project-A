# Phase 3 RAG + LangGraph Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current "dump all of bio.md into every prompt" `/chat` behavior with tenant-scoped chunked retrieval (RAG) over `bio.md`/`projects.md`/`resume.md`, orchestrated by a LangGraph classify→retrieve→generate→self-critique workflow.

**Architecture:** Markdown files are chunked on heading boundaries, embedded via Gemini, and stored in a pgvector `embeddings` table scoped by `tenant_id`. `/chat/{tenant_slug}` invokes a small LangGraph state graph that classifies the question, retrieves relevant chunks for that tenant, generates an answer, and self-critiques once before responding.

**Tech Stack:** FastAPI, SQLAlchemy 2.0, `pgvector` (Postgres extension + `pgvector-python`), `openai` SDK against Gemini's OpenAI-compatible endpoint (chat + embeddings), `langgraph`, `pytest`.

## Global Constraints

- Every `embeddings` row carries `tenant_id`; every retrieval query filters by it. Cross-tenant leakage is a correctness bug, not a cosmetic one (spec, Data model section).
- Chunking splits on markdown structural boundaries (`##`/`###` headings), not fixed character/token counts (spec, Ingestion & chunking).
- Embeddings use Gemini's embedding model via the same OpenAI-compatible client already used for chat — no new provider/API key (spec, Ingestion & chunking; Dependencies).
- Retrieval is plain cosine-similarity top-k — no hybrid search, reranking, or query rewriting in this slice (spec, Retrieval).
- The LangGraph self-critique loop retries at most once, then answers regardless — never loops indefinitely (spec, LangGraph workflow).
- The MCP tool (`fetch_github_activity`) is out of scope for this plan (spec, Scope decision; Non-goals).
- Tasks that touch the database require Postgres running: `docker compose up -d` from the repo root before running their tests.

---

### Task 1: Heading-based markdown chunker

**Files:**
- Create: `backend/app/chunking.py`
- Test: `backend/tests/test_chunking.py`

**Interfaces:**
- Produces: `Chunk` (dataclass: `source_file: str`, `chunk_index: int`, `text: str`); `chunk_markdown(text: str, source_file: str) -> list[Chunk]`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_chunking.py
from app.chunking import chunk_markdown

SAMPLE = """# Title

Intro text before any heading (should be dropped).

## Section One
Some text here.

## Section Two
More text.

### Subsection
Nested text.
"""


def test_splits_on_h2_and_h3_headings():
    chunks = chunk_markdown(SAMPLE, source_file="sample.md")
    assert len(chunks) == 3
    assert chunks[0].text.startswith("## Section One")
    assert "Some text here." in chunks[0].text
    assert chunks[1].text.startswith("## Section Two")
    assert "More text." in chunks[1].text
    assert "### Subsection" not in chunks[1].text
    assert chunks[2].text.startswith("### Subsection")
    assert "Nested text." in chunks[2].text


def test_chunk_index_and_source_file_are_set():
    chunks = chunk_markdown(SAMPLE, source_file="sample.md")
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert all(c.source_file == "sample.md" for c in chunks)


def test_no_headings_returns_no_chunks():
    assert chunk_markdown("Just plain text, no headings.", source_file="x.md") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv\Scripts\python.exe -m pytest tests/test_chunking.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError` (`app.chunking` doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/chunking.py
import re
from dataclasses import dataclass

_HEADING_PATTERN = re.compile(r"^(#{2,3}\s.*)$", re.MULTILINE)


@dataclass
class Chunk:
    source_file: str
    chunk_index: int
    text: str


def chunk_markdown(text: str, source_file: str) -> list[Chunk]:
    """Split markdown into chunks on H2/H3 heading boundaries. Text before the
    first H2/H3 (e.g. an H1 title) is dropped -- it's not a retrievable unit."""
    matches = list(_HEADING_PATTERN.finditer(text))
    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(source_file=source_file, chunk_index=i, text=chunk_text))
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv\Scripts\python.exe -m pytest tests/test_chunking.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/chunking.py backend/tests/test_chunking.py
git commit -m "feat: add heading-based markdown chunker"
```

---

### Task 2: Tenant & embedding models, DB session factory

**Files:**
- Create: `backend/app/models.py`
- Modify: `backend/app/db.py`
- Modify: `backend/app/config.py`
- Test: `backend/tests/test_models.py`

**Interfaces:**
- Consumes: none (new schema layer)
- Produces: `app.db.SessionLocal` (sessionmaker); `app.models.Tenant` (`id`, `slug`, `name`, `status`, `created_at`); `app.models.EmbeddingChunk` (`id`, `tenant_id`, `source_file`, `chunk_index`, `chunk_text`, `embedding`); `app.models.create_tables() -> None`; `config.EMBEDDING_DIMENSIONS: int`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_models.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv\Scripts\python.exe -m pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError` (`app.models` doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Add to `backend/app/config.py` (after the existing Gemini constants):

```python
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "768"))
```

Modify `backend/app/db.py`:

```python
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def check_connection() -> bool:
    """Confirm Postgres is reachable and pgvector is enabled."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        return conn.execute(text("SELECT 1")).scalar() == 1
```

Create `backend/app/models.py`:

```python
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import EMBEDDING_DIMENSIONS
from app.db import engine


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EmbeddingChunk(Base):
    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), nullable=False)
    source_file: Mapped[str] = mapped_column(String, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Dimension must match the `dimensions` param passed in embeddings.py's embed_texts call.
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=False)


def create_tables() -> None:
    Base.metadata.create_all(engine)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose up -d` (if not already running), then:
`cd backend && .venv\Scripts\python.exe -m pytest tests/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/models.py backend/app/db.py backend/app/config.py backend/tests/test_models.py
git commit -m "feat: add tenant/embedding models and DB session factory"
```

---

### Task 3: Gemini embeddings wrapper

**Files:**
- Create: `backend/app/embeddings.py`
- Modify: `backend/app/config.py`
- Test: `backend/tests/test_embeddings.py`

**Interfaces:**
- Consumes: `config.GEMINI_API_KEY`, `config.GEMINI_BASE_URL`, `config.EMBEDDING_DIMENSIONS` (from Task 2)
- Produces: `embed_texts(texts: list[str]) -> list[list[float]]`; `config.GEMINI_EMBEDDING_MODEL: str`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_embeddings.py
from types import SimpleNamespace

from app import embeddings


class _FakeEmbeddingsAPI:
    def __init__(self):
        self.last_call = None

    def create(self, **kwargs):
        self.last_call = kwargs
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in kwargs["input"]]
        )


def test_embed_texts_returns_one_vector_per_input(monkeypatch):
    fake_client = SimpleNamespace(embeddings=_FakeEmbeddingsAPI())
    monkeypatch.setattr(embeddings, "_client", fake_client)

    result = embeddings.embed_texts(["hello", "world"])

    assert result == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


def test_embed_texts_calls_configured_model_and_dimensions(monkeypatch):
    fake_client = SimpleNamespace(embeddings=_FakeEmbeddingsAPI())
    monkeypatch.setattr(embeddings, "_client", fake_client)
    monkeypatch.setattr(embeddings, "GEMINI_EMBEDDING_MODEL", "test-embed-model")
    monkeypatch.setattr(embeddings, "EMBEDDING_DIMENSIONS", 3)

    embeddings.embed_texts(["hello"])

    assert fake_client.embeddings.last_call == {
        "model": "test-embed-model",
        "input": ["hello"],
        "dimensions": 3,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv\Scripts\python.exe -m pytest tests/test_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError` (`app.embeddings` doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Add to `backend/app/config.py`:

```python
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
```

Create `backend/app/embeddings.py`:

```python
from openai import OpenAI

from app.config import EMBEDDING_DIMENSIONS, GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_EMBEDDING_MODEL

_client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = _client.embeddings.create(
        model=GEMINI_EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return [item.embedding for item in response.data]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv\Scripts\python.exe -m pytest tests/test_embeddings.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Manual smoke check (not part of automated tests)**

Real Gemini embedding calls aren't exercised by the mocked tests above — confirm the live API actually accepts `dimensions` and returns vectors of that length before Task 4 relies on it:

```bash
cd backend && .venv\Scripts\python.exe -c "from app.embeddings import embed_texts; v = embed_texts(['hello world']); print(len(v[0]))"
```

Expected: prints `768`. If it errors on the `dimensions` kwarg or prints a different length, adjust `EMBEDDING_DIMENSIONS` in `config.py`/`.env` to match what the API actually returns (and re-run Task 2's `test_models.py` after, since the column width must match).

- [ ] **Step 6: Commit**

```bash
git add backend/app/embeddings.py backend/app/config.py backend/tests/test_embeddings.py
git commit -m "feat: add Gemini embeddings wrapper"
```

---

### Task 4: Tenant content layout + ingestion script

**Files:**
- Create: `backend/app/ingest.py`
- Move: `docs/content/bio.md` → `docs/content/tenants/toua/bio.md`
- Move: `docs/content/projects.md` → `docs/content/tenants/toua/projects.md`
- Move: `docs/content/resume.md` → `docs/content/tenants/toua/resume.md`
- Test: `backend/tests/test_ingest.py`

**Interfaces:**
- Consumes: `chunk_markdown` (Task 1), `embed_texts` (Task 3), `Tenant`/`EmbeddingChunk`/`create_tables`/`SessionLocal` (Task 2), `config.CONTENT_DIR`
- Produces: `tenant_content_dir(slug: str) -> Path`; `get_or_create_tenant(session, slug: str) -> Tenant`; `ingest_tenant(slug: str) -> int` (returns chunk count)

- [ ] **Step 1: Move the content files into the tenant layout**

```bash
mkdir -p docs/content/tenants/toua
git mv docs/content/bio.md docs/content/tenants/toua/bio.md
git mv docs/content/projects.md docs/content/tenants/toua/projects.md
git mv docs/content/resume.md docs/content/tenants/toua/resume.md
```

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_ingest.py
"""Requires Postgres running: `docker compose up -d` from the repo root."""
from app import ingest
from app.db import SessionLocal
from app.models import EmbeddingChunk, Tenant

TEST_SLUG = "test-ingest-tenant"


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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && .venv\Scripts\python.exe -m pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError` (`app.ingest` doesn't exist yet).

- [ ] **Step 4: Write minimal implementation**

```python
# backend/app/ingest.py
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && .venv\Scripts\python.exe -m pytest tests/test_ingest.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add docs/content/tenants backend/app/ingest.py backend/tests/test_ingest.py
git commit -m "feat: add tenant content layout and ingestion script"
```

---

### Task 5: Tenant-scoped retrieval

**Files:**
- Create: `backend/app/retrieval.py`
- Modify: `backend/app/config.py`
- Test: `backend/tests/test_retrieval.py`

**Interfaces:**
- Consumes: `Tenant`/`EmbeddingChunk`/`SessionLocal`/`create_tables` (Task 2)
- Produces: `retrieve_chunks(tenant_id: int, query_embedding: list[float], k: int = RETRIEVAL_TOP_K) -> list[str]`; `config.RETRIEVAL_TOP_K: int`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_retrieval.py
"""Requires Postgres running: `docker compose up -d` from the repo root."""
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


def test_retrieval_never_crosses_tenants():
    _cleanup()
    tenant_a_id, tenant_b_id = _seed()
    try:
        results = retrieve_chunks(tenant_a_id, _vector(0), k=5)
        assert all("tenant A" in r for r in results)
        assert not any("tenant B" in r for r in results)
    finally:
        _cleanup()


def test_retrieval_orders_by_similarity():
    _cleanup()
    tenant_a_id, _ = _seed()
    try:
        results = retrieve_chunks(tenant_a_id, _vector(0), k=1)
        assert results == ["tenant A chunk 0"]
    finally:
        _cleanup()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv\Scripts\python.exe -m pytest tests/test_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError` (`app.retrieval` doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Add to `backend/app/config.py`:

```python
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "5"))
```

Create `backend/app/retrieval.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv\Scripts\python.exe -m pytest tests/test_retrieval.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/retrieval.py backend/app/config.py backend/tests/test_retrieval.py
git commit -m "feat: add tenant-scoped vector retrieval"
```

---

### Task 6: LangGraph classify→retrieve→generate→self-critique workflow

**Files:**
- Create: `backend/app/workflow.py`
- Modify: `backend/requirements.txt`
- Test: `backend/tests/test_workflow.py`

**Interfaces:**
- Consumes: `embed_texts` (Task 3), `retrieve_chunks` (Task 5), `config.GEMINI_API_KEY`/`GEMINI_BASE_URL`/`GEMINI_MODEL`
- Produces: `run_chat_workflow(tenant_id: int, question: str) -> str`

- [ ] **Step 1: Add the dependency**

Add to `backend/requirements.txt`:

```
langgraph>=0.2
```

Run: `cd backend && .venv\Scripts\python.exe -m pip install -r requirements.txt`

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_workflow.py
from types import SimpleNamespace

from app import workflow


class _ScriptedChatAPI:
    """Returns queued responses in order; each call consumes the next one."""

    def __init__(self, contents: list[str]):
        self._queue = list(contents)
        self.call_count = 0

    def create(self, **kwargs):
        self.call_count += 1
        content = self._queue.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _patch_common(monkeypatch, chat_contents: list[str]):
    fake_chat_api = _ScriptedChatAPI(chat_contents)
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_chat_api))
    monkeypatch.setattr(workflow, "_client", fake_client)
    monkeypatch.setattr(workflow, "embed_texts", lambda texts: [[0.0, 0.0, 0.0] for _ in texts])
    monkeypatch.setattr(workflow, "retrieve_chunks", lambda tenant_id, vec, **kw: ["some background chunk"])
    return fake_chat_api


def test_answers_directly_when_critique_passes(monkeypatch):
    # order: classify, generate, critique
    fake_chat_api = _patch_common(monkeypatch, ["general", "first answer", "pass"])

    answer = workflow.run_chat_workflow(tenant_id=1, question="What do you do?")

    assert answer == "first answer"
    assert fake_chat_api.call_count == 3


def test_retries_exactly_once_when_critique_fails(monkeypatch):
    # order: classify, generate, critique(fail), generate(retry), (critique skipped on retry)
    fake_chat_api = _patch_common(
        monkeypatch, ["background", "first answer", "fail", "second answer"]
    )

    answer = workflow.run_chat_workflow(tenant_id=1, question="Where did you go to school?")

    assert answer == "second answer"
    assert fake_chat_api.call_count == 4
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && .venv\Scripts\python.exe -m pytest tests/test_workflow.py -v`
Expected: FAIL with `ModuleNotFoundError` (`app.workflow` doesn't exist yet).

- [ ] **Step 4: Write minimal implementation**

```python
# backend/app/workflow.py
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from app.config import GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_MODEL
from app.embeddings import embed_texts
from app.retrieval import retrieve_chunks

_client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)


class ChatState(TypedDict):
    tenant_id: int
    question: str
    query: str
    category: str
    chunks: list[str]
    answer: str
    retry_used: bool
    needs_retry: bool


def _classify_node(state: ChatState) -> dict:
    completion = _client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the user's question into exactly one category: "
                    "'background', 'project', or 'general'. Respond with only that word."
                ),
            },
            {"role": "user", "content": state["question"]},
        ],
    )
    category = (completion.choices[0].message.content or "general").strip().lower()
    if category not in ("background", "project", "general"):
        category = "general"
    return {"category": category}


def _retrieve_node(state: ChatState) -> dict:
    [query_vector] = embed_texts([state["query"]])
    chunks = retrieve_chunks(state["tenant_id"], query_vector)
    return {"chunks": chunks}


def _generate_node(state: ChatState) -> dict:
    context = "\n\n".join(state["chunks"]) or "(no matching background information found)"
    completion = _client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI assistant speaking on behalf of the person described "
                    "below. Answer in first person, grounded only in the provided "
                    "background, and say when something isn't covered by it.\n\n"
                    f"--- BACKGROUND ---\n{context}"
                ),
            },
            {"role": "user", "content": state["question"]},
        ],
    )
    return {"answer": completion.choices[0].message.content or ""}


def _critique_node(state: ChatState) -> dict:
    if state["retry_used"]:
        return {"needs_retry": False}

    completion = _client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Judge whether the ANSWER is grounded in the CONTEXT and actually "
                    "addresses the QUESTION. Respond with only 'pass' or 'fail'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION: {state['question']}\n\n"
                    f"CONTEXT: {chr(10).join(state['chunks'])}\n\n"
                    f"ANSWER: {state['answer']}"
                ),
            },
        ],
    )
    verdict = (completion.choices[0].message.content or "pass").strip().lower()
    if verdict.startswith("fail"):
        return {
            "needs_retry": True,
            "retry_used": True,
            "query": f"{state['question']} (be more specific and grounded)",
        }
    return {"needs_retry": False}


def _route_after_critique(state: ChatState) -> str:
    return "retrieve" if state["needs_retry"] else END


def _build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("classify", _classify_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("generate", _generate_node)
    graph.add_node("critique", _critique_node)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges("critique", _route_after_critique, {"retrieve": "retrieve", END: END})
    return graph.compile()


_graph = _build_graph()


def run_chat_workflow(tenant_id: int, question: str) -> str:
    result = _graph.invoke(
        {
            "tenant_id": tenant_id,
            "question": question,
            "query": question,
            "category": "",
            "chunks": [],
            "answer": "",
            "retry_used": False,
            "needs_retry": False,
        }
    )
    return result["answer"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && .venv\Scripts\python.exe -m pytest tests/test_workflow.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add backend/app/workflow.py backend/requirements.txt backend/tests/test_workflow.py
git commit -m "feat: add LangGraph classify/retrieve/generate/self-critique workflow"
```

---

### Task 7: Wire `/chat/{tenant_slug}` in main.py

**Files:**
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_chat_endpoint.py`

**Interfaces:**
- Consumes: `run_chat_workflow` (Task 6), `Tenant`/`EmbeddingChunk`/`SessionLocal`/`create_tables` (Task 2), `auth.require_user`, `ratelimit.enforce_chat_rate_limit` (existing)
- Produces: `POST /chat/{tenant_slug}` route (replaces `POST /chat`)

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_chat_endpoint.py
"""Requires Postgres running: `docker compose up -d` from the repo root."""
import time

from fastapi.testclient import TestClient

from app import main
from app.auth import require_user
from app.db import SessionLocal
from app.models import EmbeddingChunk, Tenant, create_tables

FAKE_USER = {"email": "chat@example.com", "name": "Chat", "exp": time.time() + 3600}
TENANT_WITH_CONTENT = "test-chat-tenant-with-content"
TENANT_NO_CONTENT = "test-chat-tenant-empty"


def _cleanup():
    session = SessionLocal()
    try:
        for slug in (TENANT_WITH_CONTENT, TENANT_NO_CONTENT):
            tenant = session.query(Tenant).filter_by(slug=slug).one_or_none()
            if tenant is not None:
                session.query(EmbeddingChunk).filter_by(tenant_id=tenant.id).delete()
                session.query(Tenant).filter_by(id=tenant.id).delete()
        session.commit()
    finally:
        session.close()


def _client():
    main.app.dependency_overrides[require_user] = lambda: FAKE_USER
    return TestClient(main.app)


def test_unknown_tenant_returns_404():
    resp = _client().post("/chat/does-not-exist", json={"message": "hi"})
    assert resp.status_code == 404


def test_tenant_with_no_content_returns_explicit_message(monkeypatch):
    _cleanup()
    create_tables()
    session = SessionLocal()
    try:
        session.add(Tenant(slug=TENANT_NO_CONTENT, name="Empty", status="active"))
        session.commit()
    finally:
        session.close()

    try:
        resp = _client().post(f"/chat/{TENANT_NO_CONTENT}", json={"message": "hi"})
        assert resp.status_code == 200
        assert "don't have any information" in resp.json()["reply"]
    finally:
        _cleanup()


def test_tenant_with_content_returns_workflow_answer(monkeypatch):
    _cleanup()
    create_tables()
    session = SessionLocal()
    try:
        tenant = Tenant(slug=TENANT_WITH_CONTENT, name="Has Content", status="active")
        session.add(tenant)
        session.flush()
        session.add(
            EmbeddingChunk(
                tenant_id=tenant.id, source_file="bio.md", chunk_index=0,
                chunk_text="chunk", embedding=[0.0] * 768,
            )
        )
        session.commit()
    finally:
        session.close()

    monkeypatch.setattr(main.workflow, "run_chat_workflow", lambda tenant_id, question: "mocked answer")

    try:
        resp = _client().post(f"/chat/{TENANT_WITH_CONTENT}", json={"message": "hi"})
        assert resp.status_code == 200
        assert resp.json()["reply"] == "mocked answer"
    finally:
        _cleanup()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv\Scripts\python.exe -m pytest tests/test_chat_endpoint.py -v`
Expected: FAIL — route `/chat/{tenant_slug}` doesn't exist yet (404s or 405s where the test expects 200, or fails locating `main.workflow`).

- [ ] **Step 3: Modify `backend/app/main.py`**

Replace the imports, module-level client, `build_system_prompt`, and `/chat` route with:

```python
import logging
import time

from fastapi import Depends, FastAPI, HTTPException
from openai import OpenAIError
from pydantic import BaseModel
from sqlalchemy import select
from starlette.middleware.sessions import SessionMiddleware

from app import auth, ratelimit, workflow
from app.config import (
    SESSION_COOKIE_SECURE,
    SESSION_MAX_AGE_SECONDS,
    SESSION_SECRET,
)
from app.db import SessionLocal
from app.models import EmbeddingChunk, Tenant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("askme")

app = FastAPI(title="Ask Me")

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    max_age=SESSION_MAX_AGE_SECONDS,
    same_site="lax",
    https_only=SESSION_COOKIE_SECURE,
)

app.include_router(auth.router)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat/{tenant_slug}", response_model=ChatResponse)
def chat(
    tenant_slug: str,
    req: ChatRequest,
    user: dict = Depends(auth.require_user),
    _: None = Depends(ratelimit.enforce_chat_rate_limit),
) -> ChatResponse:
    session = SessionLocal()
    try:
        tenant = session.scalar(select(Tenant).where(Tenant.slug == tenant_slug))
        if tenant is None:
            raise HTTPException(status_code=404, detail="Unknown tenant.")

        has_content = session.scalar(
            select(EmbeddingChunk.id).where(EmbeddingChunk.tenant_id == tenant.id).limit(1)
        )
        if has_content is None:
            return ChatResponse(reply="I don't have any information loaded yet -- check back soon.")
        tenant_id = tenant.id
    finally:
        session.close()

    start = time.monotonic()
    try:
        answer = workflow.run_chat_workflow(tenant_id, req.message)
    except OpenAIError:
        logger.exception("Gemini request failed for user=%s", user["email"])
        raise HTTPException(
            status_code=502,
            detail="Could not reach the assistant right now — try again shortly.",
        )

    logger.info(
        "chat request user=%s tenant=%s elapsed=%.2fs", user["email"], tenant_slug, time.monotonic() - start
    )
    return ChatResponse(reply=answer)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose up -d` (if not already running), then:
`cd backend && .venv\Scripts\python.exe -m pytest tests/test_chat_endpoint.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full backend test suite**

Run: `cd backend && .venv\Scripts\python.exe -m pytest -v`
Expected: PASS (all tests, including Tasks 1–6 and the pre-existing `test_auth.py`/`test_ratelimit.py`)

- [ ] **Step 6: Commit**

```bash
git add backend/app/main.py backend/tests/test_chat_endpoint.py
git commit -m "feat: wire tenant-scoped /chat/{tenant_slug} to the RAG workflow"
```

---

### Task 8: Update the frontend widget and setup docs

**Files:**
- Modify: `frontend/src/ChatWidget.jsx`
- Modify: `README.md`

**Interfaces:**
- Consumes: `POST /chat/{tenant_slug}` (Task 7)

- [ ] **Step 1: Point the widget at the tenant-scoped endpoint**

In `frontend/src/ChatWidget.jsx`, add a constant near the top of the file and use it in the `fetch` call:

```jsx
const TENANT_SLUG = "toua";
```

Change:

```jsx
const res = await fetch("/chat", {
```

to:

```jsx
const res = await fetch(`/chat/${TENANT_SLUG}`, {
```

- [ ] **Step 2: Update `README.md` setup instructions**

In the "Setup" section, after the existing step 3 (Backend), add an ingestion step and update the `curl` verification example:

```markdown
### 4. Ingest your content

```bash
cd backend
.venv\Scripts\python.exe -m app.ingest toua
```

Re-run this any time you edit `docs/content/tenants/toua/*.md`.
```

Update the existing `curl` example from `POST http://localhost:8000/chat` to `POST http://localhost:8000/chat/toua`. Renumber the subsequent "Frontend" step accordingly. Update the "Project structure" section's `docs/content/` line to note the `tenants/<slug>/` layout.

- [ ] **Step 3: Manual verification (no frontend test suite exists in this repo)**

```bash
docker compose up -d
cd backend && .venv\Scripts\python.exe -m app.ingest toua
cd backend && .venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

In a second terminal:

```bash
cd frontend && npm run dev
```

Open `http://localhost:5173`, sign in, and ask a question the bio actually answers (e.g. "Where did you go to school?"). Confirm:
- The reply is grounded in real content from `bio.md` (not a generic/empty answer).
- The browser network tab shows the request going to `/chat/toua` and returning 200.
- Asking something clearly outside the bio (e.g. "What's your favorite pizza topping?") gets an answer that says it isn't covered, rather than a hallucinated one.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/ChatWidget.jsx README.md
git commit -m "feat: point chat widget at tenant-scoped endpoint, update setup docs"
```

---

## Self-Review Notes

- **Spec coverage:** data model (Task 2), ingestion/chunking (Tasks 1, 4), retrieval (Task 5), LangGraph workflow (Task 6), API & error handling (Task 7), embeddings/dependencies (Tasks 3, 6). Testing section's isolation test (Task 5), chunking sanity test (Task 1), self-critique retry-cap test (Task 6) are all covered. The retrieval relevance spot-check from the spec is covered informally by Task 8's manual verification rather than an automated test, since it's explicitly scoped as informal in the spec.
- **Deferred by design (per spec Non-goals):** MCP tool, hybrid search/reranking, auto-reingestion, self-serve tenant signup — none of these appear as tasks, intentionally.
- **Type/signature consistency checked:** `Chunk` (Task 1) fields match how Task 4 constructs `EmbeddingChunk` rows; `EMBEDDING_DIMENSIONS` (Task 2) matches the `dimensions` arg in `embed_texts` (Task 3) and the test vectors in Tasks 2, 4, 5, 7; `retrieve_chunks` signature (Task 5) matches its call in `workflow.py` (Task 6); `run_chat_workflow(tenant_id, question)` (Task 6) matches its call in `main.py` (Task 7).
