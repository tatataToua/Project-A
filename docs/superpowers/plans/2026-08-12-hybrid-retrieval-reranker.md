# Hybrid Retrieval + Reranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace pure pgvector cosine-distance retrieval with hybrid retrieval (pgvector
semantic search + Postgres full-text search, fused via reciprocal rank fusion) followed
by cross-encoder reranking — closing the single biggest "demo vs. production RAG" gap
identified in the Phase 3.5 design spec.

**Architecture:** `retrieval.py`'s `retrieve_chunks` becomes a three-stage pipeline:
over-fetch candidates from pgvector and from a new Postgres full-text (`tsvector`)
column in parallel, fuse the two ranked lists with reciprocal rank fusion, then rescore
the fused candidates with a local cross-encoder (`app/reranking.py`, new module) down to
the final top-k. `workflow.py`'s `_retrieve_node` passes both the raw query text and its
embedding through (previously only the embedding). Tenant isolation is preserved: every
new query path filters by `tenant_id`, matching the existing correctness requirement.

**Tech Stack:** SQLAlchemy Core (raw SQL for `tsvector`/`ts_rank`, since the new column
isn't ORM-mapped), Postgres `GENERATED ALWAYS AS ... STORED` for the full-text column
(no backfill step needed — Postgres computes it for existing rows when the column is
added), `sentence-transformers` (new dependency) for the cross-encoder.

## Global Constraints

- Every retrieval query must filter by `tenant_id` — this is the tenant isolation
  boundary and a correctness requirement, not a later add-on (per `CLAUDE.md` and the
  SaaS design spec).
- No Alembic/migration tooling in this repo — schema changes are handled as idempotent
  raw SQL run from `create_tables()` (`models.py`), consistent with how
  `EMBEDDING_DIMENSIONS` changes are already documented as a manual step.
- BM25-style search uses Postgres full-text search (`tsvector`/`ts_rank`), not a
  separate search engine or the `rank_bm25` package — decided in the Phase 3.5 spec to
  avoid a new dependency and stay in the DB layer beside pgvector.
- Reranker is a local `sentence-transformers` cross-encoder (accepted `torch`
  dependency), not an external API (e.g. Cohere) or an LLM-prompted reranker — decided
  in the Phase 3.5 spec.
- Windows dev environment: run backend commands via `.venv\Scripts\python.exe` /
  `.venv\Scripts\python.exe -m pytest` from `backend/`, with Postgres up
  (`docker compose up -d` from repo root).
- RRF fusion constant is a fixed internal implementation detail (60, the standard
  smoothing constant from the literature), not an operator-tunable config value.

---

## Task 1: Full-text search column + index

**Files:**
- Modify: `backend/app/models.py`
- Test: `backend/tests/test_models.py`

**Interfaces:**
- Produces: `create_tables()` (existing function, unchanged signature) now also ensures
  `embeddings.chunk_tsv` (a Postgres `tsvector` generated column) and a GIN index on it
  exist. Idempotent — safe to call on every app startup, exactly like today.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_models.py` (after the existing `test_create_tenant_and_embedding_chunk`):

```python
from sqlalchemy import text


def test_create_tables_enables_fulltext_search_on_chunk_text():
    create_tables()
    _cleanup()
    session = SessionLocal()
    try:
        tenant = Tenant(slug=TEST_SLUG, name="Test Tenant", status="active")
        session.add(tenant)
        session.flush()
        session.add(
            EmbeddingChunk(
                tenant_id=tenant.id,
                source_file="bio.md",
                chunk_index=0,
                chunk_text="The tavern serves craft beer on tap every evening.",
                embedding=_vector(2),
            )
        )
        session.commit()

        row = session.execute(
            text(
                "SELECT chunk_text FROM embeddings "
                "WHERE tenant_id = :tid AND chunk_tsv @@ plainto_tsquery('english', :q)"
            ),
            {"tid": tenant.id, "q": "craft beer"},
        ).one_or_none()

        assert row is not None
        assert "craft beer" in row.chunk_text
    finally:
        session.close()
        _cleanup()
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `backend/`): `.venv\Scripts\python.exe -m pytest tests/test_models.py::test_create_tables_enables_fulltext_search_on_chunk_text -v`
Expected: FAIL with a Postgres error — `column "chunk_tsv" does not exist` (via
`psycopg.errors.UndefinedColumn`).

- [ ] **Step 3: Implement the migration step**

In `backend/app/models.py`, change the import line:

```python
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
```

to:

```python
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func, text
```

Then add a new function and call it from `create_tables()`:

```python
def _ensure_fulltext_search() -> None:
    """Add the tsvector column + GIN index that back hybrid (BM25-style)
    retrieval, if they don't already exist.

    `Base.metadata.create_all()` never alters an existing table, so a database
    that already has an `embeddings` table from before this change needs this
    explicit step. It's a Postgres GENERATED ALWAYS AS ... STORED column, so
    Postgres computes `chunk_tsv` for every existing row as part of the ALTER
    TABLE itself -- no separate backfill/re-ingest step needed. Idempotent via
    IF NOT EXISTS, so safe to run on every `create_tables()` call.
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                ALTER TABLE embeddings
                ADD COLUMN IF NOT EXISTS chunk_tsv tsvector
                GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED
                """
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS embeddings_chunk_tsv_idx "
                "ON embeddings USING GIN (chunk_tsv)"
            )
        )


def create_tables() -> None:
    # `create_all` only creates missing tables — it never migrates an existing one.
    # If you change EMBEDDING_DIMENSIONS after the `embeddings` table already exists,
    # this call silently no-ops and the old vector width stays in place; you must
    # `DROP TABLE embeddings;` and re-run ingestion, or inserts will fail later with a
    # confusing dimension-mismatch error that points nowhere near the config change.
    #
    # Ensure pgvector's `vector` type exists before creating tables that use it —
    # required on a fresh database where the extension hasn't been enabled yet.
    check_connection()
    Base.metadata.create_all(engine)
    _ensure_fulltext_search()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_models.py -v`
Expected: all tests in the file PASS, including the new one.

- [ ] **Step 5: Commit**

```bash
git add backend/app/models.py backend/tests/test_models.py
git commit -m "Add Postgres full-text search column/index for hybrid retrieval"
```

---

## Task 2: Cross-encoder reranker module

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/app/config.py`
- Create: `backend/app/reranking.py`
- Test: `backend/tests/test_reranking.py`

**Interfaces:**
- Consumes: nothing from Task 1 (independent unit).
- Produces: `rerank_chunks(query_text: str, candidates: list[tuple[int, str]], k: int) -> list[str]`
  in `app/reranking.py` — takes `(chunk_id, chunk_text)` candidates, returns just the
  `chunk_text` values of the top-k, best first. Consumed by Task 3.

- [ ] **Step 1: Add the dependency and install it**

In `backend/requirements.txt`, add after the `langgraph` line:

```
sentence-transformers>=3.0
```

Run (from `backend/`, venv active): `.venv\Scripts\python.exe -m pip install -r requirements.txt`

This pulls in `torch` — expect a multi-minute install the first time.

- [ ] **Step 2: Add reranker config**

In `backend/app/config.py`, after the `RETRIEVAL_TOP_K` line, add:

```python
RETRIEVAL_OVERFETCH_K = int(os.environ.get("RETRIEVAL_OVERFETCH_K", "15"))
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
```

- [ ] **Step 3: Write the failing tests**

Create `backend/tests/test_reranking.py`:

```python
"""First real-model test downloads cross-encoder/ms-marco-MiniLM-L-6-v2 on first
run (network + local Hugging Face cache) -- subsequent runs are fast."""
from app import reranking


def test_rerank_chunks_returns_empty_list_for_no_candidates():
    assert reranking.rerank_chunks("query", [], k=5) == []


def test_rerank_chunks_orders_by_score_descending(monkeypatch):
    class _FakeModel:
        def predict(self, pairs):
            return [1.0 if "match" in chunk_text else 0.0 for _q, chunk_text in pairs]

    monkeypatch.setattr(reranking, "_get_model", lambda: _FakeModel())

    candidates = [
        (1, "irrelevant text"),
        (2, "this is a match"),
        (3, "also irrelevant"),
    ]
    result = reranking.rerank_chunks("query", candidates, k=2)

    assert result == ["this is a match", "irrelevant text"]


def test_rerank_chunks_truncates_to_k(monkeypatch):
    class _FakeModel:
        def predict(self, pairs):
            return [float(i) for i in range(len(pairs))]

    monkeypatch.setattr(reranking, "_get_model", lambda: _FakeModel())

    candidates = [(1, "a"), (2, "b"), (3, "c")]
    result = reranking.rerank_chunks("query", candidates, k=1)

    assert result == ["c"]


def test_rerank_chunks_uses_the_real_cross_encoder_model():
    candidates = [
        (1, "The restaurant is open from 5pm to 10pm on weekdays."),
        (2, "Our chef trained in Paris for six years."),
    ]
    result = reranking.rerank_chunks("What time do you open?", candidates, k=1)

    assert result == ["The restaurant is open from 5pm to 10pm on weekdays."]
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_reranking.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'app.reranking'`.

- [ ] **Step 5: Implement `app/reranking.py`**

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_reranking.py -v`
Expected: all 4 tests PASS (the last one takes longer the first time — model download).

- [ ] **Step 7: Commit**

```bash
git add backend/requirements.txt backend/app/config.py backend/app/reranking.py backend/tests/test_reranking.py
git commit -m "Add cross-encoder reranker module"
```

---

## Task 3: Hybrid retrieval + wiring into the workflow

**Files:**
- Modify: `backend/app/retrieval.py`
- Modify: `backend/app/workflow.py`
- Modify: `backend/tests/test_retrieval.py`
- Modify: `backend/tests/test_workflow.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `rerank_chunks` from `app/reranking.py` (Task 2); the `chunk_tsv` column
  from `app/models.py` (Task 1).
- Produces: `retrieve_chunks(tenant_id: int, query_text: str, query_embedding:
  list[float], k: int = RETRIEVAL_TOP_K, overfetch_k: int = RETRIEVAL_OVERFETCH_K) ->
  list[str]` — **signature change** from the current `(tenant_id, query_embedding, k)`.
  Every caller must pass the raw query text as the second positional argument now.

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `backend/tests/test_retrieval.py`:

```python
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
    hybrid results already arrived in, just truncating to k. Used so these
    tests exercise the SQL/RRF logic in isolation, deterministically, without
    depending on the (slower, model-download-requiring) real reranker."""
    return [chunk_text for _chunk_id, chunk_text in candidates[:k]]


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
        assert all("tenant A" in r for r in results)
        assert not any("tenant B" in r for r in results)
    finally:
        _cleanup()


def test_retrieval_orders_by_similarity_when_no_keyword_signal(monkeypatch):
    monkeypatch.setattr(retrieval, "rerank_chunks", _passthrough_rerank)
    _cleanup()
    tenant_a_id, _ = _seed()
    try:
        # "xyz" matches no chunk's text, so full-text search contributes nothing
        # to the fusion -- ordering comes entirely from vector similarity.
        results = retrieve_chunks(tenant_a_id, "xyz", _vector(0), k=1)
        assert results == ["tenant A chunk 0"]
    finally:
        _cleanup()


def test_fulltext_signal_can_outrank_pure_vector_similarity(monkeypatch):
    """A chunk with a strong keyword match but a vector far from the query
    embedding still wins the top spot once fused with a chunk that only has
    vector similarity going for it -- proving hybrid fusion actually combines
    both signals rather than one silently dominating."""
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
                    embedding=_vector(500),  # orthogonal to the query vector below
                ),
            ]
        )
        session.commit()
        tenant_id = tenant.id
    finally:
        session.close()

    try:
        # Query vector exactly matches chunk 0's embedding, so pure vector
        # search alone would return chunk 0 first. The karaoke chunk only
        # wins because it also matches the full-text query.
        results = retrieve_chunks(tenant_id, "karaoke night", _vector(0), k=1)
        assert results == ["the tavern has a dedicated karaoke night"]
    finally:
        _cleanup()


def test_reranker_is_actually_invoked_end_to_end():
    """No monkeypatch here -- proves the real cross-encoder is wired into
    retrieve_chunks, not just the passthrough used above. Downloads the model
    on first run (network + local cache)."""
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
                    chunk_text="We are open from 5pm to 10pm Tuesday through Sunday.",
                    embedding=_vector(0),
                ),
                EmbeddingChunk(
                    tenant_id=tenant.id, source_file="bio.md", chunk_index=1,
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
        assert results == ["We are open from 5pm to 10pm Tuesday through Sunday."]
    finally:
        _cleanup()
```

Also update `backend/tests/test_workflow.py`'s `_patch_common` (the monkeypatched
`retrieve_chunks` lambda needs to accept the new `query_text` argument):

Change:
```python
    monkeypatch.setattr(workflow, "retrieve_chunks", lambda tenant_id, vec, **kw: ["some background chunk"])
```
to:
```python
    monkeypatch.setattr(workflow, "retrieve_chunks", lambda tenant_id, query_text, vec, **kw: ["some background chunk"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_retrieval.py tests/test_workflow.py -v`
Expected: `test_retrieval.py` tests FAIL with `TypeError: retrieve_chunks() takes from 2
to 3 positional arguments but 4 were given` (or similar — old signature doesn't accept
`query_text`). `test_workflow.py` tests should still pass at this point (the lambda
change is forward-compatible prep, not yet exercised differently) — confirm they do.

- [ ] **Step 3: Implement hybrid retrieval in `app/retrieval.py`**

Replace the full contents of `backend/app/retrieval.py`:

```python
from sqlalchemy import select, text

from app.config import RETRIEVAL_OVERFETCH_K, RETRIEVAL_TOP_K
from app.db import SessionLocal
from app.models import EmbeddingChunk
from app.reranking import rerank_chunks

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
    session = SessionLocal()
    try:
        vector_hits = _vector_search(session, tenant_id, query_embedding, overfetch_k)
        fulltext_hits = _fulltext_search(session, tenant_id, query_text, overfetch_k)
        fused = _reciprocal_rank_fusion([vector_hits, fulltext_hits], k=overfetch_k)
        return rerank_chunks(query_text, fused, k)
    finally:
        session.close()
```

- [ ] **Step 4: Wire the query text through in `app/workflow.py`**

In `backend/app/workflow.py`, change `_retrieve_node`:

```python
def _retrieve_node(state: ChatState) -> dict:
    search_text = state["query"]
    if state["category"]:
        search_text = f"[{state['category']}] {search_text}"
    [query_vector] = embed_texts([search_text])
    chunks = retrieve_chunks(state["tenant_id"], search_text, query_vector)
    return {"chunks": chunks}
```

(the only change is the added `search_text` argument on the `retrieve_chunks` call).

- [ ] **Step 5: Document the two new env vars in `CLAUDE.md`**

In `CLAUDE.md`'s "Required environment variables" table, add two rows after the
`RETRIEVAL_TOP_K` row:

```markdown
| `RETRIEVAL_OVERFETCH_K` | Candidates fetched per branch (vector + full-text) before fusion/reranking, defaults to `15` |
| `RERANK_MODEL` | Cross-encoder model for reranking, defaults to `cross-encoder/ms-marco-MiniLM-L-6-v2` |
```

- [ ] **Step 6: Run the full backend test suite**

Run: `.venv\Scripts\python.exe -m pytest -v`
Expected: all tests PASS (existing suite + new retrieval/reranking/workflow tests).

- [ ] **Step 7: Commit**

```bash
git add backend/app/retrieval.py backend/app/workflow.py backend/tests/test_retrieval.py backend/tests/test_workflow.py CLAUDE.md
git commit -m "Wire hybrid retrieval + reranker into the chat workflow"
```

---

## Task 4: Manual end-to-end verification

No code changes — this task confirms the happy path still works against real content
and a real LLM/embedding provider, per the Phase 3.5 spec's success criteria. Nothing
to commit.

- [x] **Step 1: Re-ingest the demo tenant** (picks up the new full-text column via the
  existing ingest path — no ingest.py changes were needed since the column is
  Postgres-generated)

Run (from `backend/`, Postgres up via `docker compose up -d` from repo root, `.env`
configured with either Gemini or Ollama):

```bash
.venv\Scripts\python.exe -m app.ingest two-owls-tavern
```

Expected: prints `Ingested N chunks for tenant 'two-owls-tavern'` with no errors.

Result: `Ingested 34 chunks for tenant 'two-owls-tavern'`.

- [x] **Step 2: Run a real chat round-trip**

Start the backend (`uvicorn app.main:app --reload --port 8000` from `backend/`) and the
frontend (`npm run dev` from `frontend/`), then open `http://localhost:5173`, sign in,
and ask a couple of real questions (e.g. "What are your hours?", "Do you have a
karaoke night?" if that's in the demo content, or any menu/policy question).

Expected: answers return normally, grounded in the tenant's content, no errors in
either terminal.

Result: verified via `python -m app.trace_chat two-owls-tavern` (runs the real graph
against the real Gemini/embedding provider and reranker, bypassing only the browser
OAuth login, not the workflow) instead of the browser — "What are your hours?" and "Do
you have any gluten-free menu options?" both answered correctly, grounded in retrieved
chunks, first-pass critique, no errors.

- [x] **Step 3: Confirm the full test suite still passes**

Run (from `backend/`): `.venv\Scripts\python.exe -m pytest -v`
Expected: all tests PASS. Report the final count (e.g. "38/38 passing") back — this
number replaces the "29/29 passing" baseline in `METRICS.md` next time that ledger is
updated (not part of this plan — `METRICS.md` updates are their own future step).

Result: 41/41 passing.
