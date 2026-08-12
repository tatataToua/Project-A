# Citation Enforcement + Refusal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make retrieved chunks carry source attribution end to end, and make the
existing self-critique retry loop actually enforce groundedness — if a retried answer
still fails the critique, the workflow returns an explicit decline instead of an answer
that might not be grounded. This is sub-project #3 of the Phase 3.5 design spec.

**Architecture:** `retrieve_chunks` (and everything upstream of it — vector search,
full-text search, RRF fusion, reranking) changes its unit of currency from a bare
`chunk_text` string to a `(source_file, chunk_text)` tuple, so the generate node can
attribute context to a source. `_critique_node` in `workflow.py` changes from
short-circuiting (skipping the LLM judgment entirely) on a second pass to actually
re-judging the retried answer — if it still fails, the node overwrites `answer` with a
fixed decline message instead of letting an ungrounded answer through. `tracing.py`'s
per-node preview logic is updated to reflect the new three-way critique outcome
(pass / retrying / declined) instead of the old two-way (skip-detecting) logic.

**Tech Stack:** No new dependencies — pure refactor of existing SQLAlchemy queries and
LangGraph node logic.

## Global Constraints

- Every retrieval query must still filter by `tenant_id` — unaffected by this change,
  but no task may loosen it while touching these queries.
- No new dependencies.
- Windows dev environment: run backend commands via the venv at
  `C:\pyvenvs\p35backend\Scripts\python.exe -m pytest` (this worktree's venv lives
  outside the repo due to a Windows MAX_PATH issue with `torch`'s nested license
  files under the repo's deeply-nested worktree path — do not `pip install` into a
  `backend\.venv` inside this worktree).
- Postgres is already running (`docker compose up -d` from the repo root) and the
  `two-owls-tavern` tenant is already ingested.

---

## Task 1: Carry source attribution through retrieval and reranking

**Files:**
- Modify: `backend/app/retrieval.py`
- Modify: `backend/app/reranking.py`
- Modify: `backend/tests/test_retrieval.py`
- Modify: `backend/tests/test_reranking.py`

**Interfaces:**
- Produces: `retrieve_chunks(tenant_id, query_text, query_embedding, k=RETRIEVAL_TOP_K,
  overfetch_k=RETRIEVAL_OVERFETCH_K) -> list[tuple[str, str]]` — **return type change**
  from `list[str]` to `list[tuple[str, str]]` of `(source_file, chunk_text)`, best
  first. Consumed by Task 2's `workflow.py` changes.
- Produces: `rerank_chunks(query_text: str, candidates: list[tuple[int, str, str]], k:
  int) -> list[tuple[str, str]]` in `app/reranking.py` — **signature change**: candidates
  are now `(chunk_id, source_file, chunk_text)` triples (was `(chunk_id, chunk_text)`
  pairs), and the return type is `(source_file, chunk_text)` tuples (was bare
  `chunk_text` strings).

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `backend/tests/test_reranking.py`:

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
        (1, "a.md", "irrelevant text"),
        (2, "b.md", "this is a match"),
        (3, "c.md", "also irrelevant"),
    ]
    result = reranking.rerank_chunks("query", candidates, k=2)

    assert result == [("b.md", "this is a match"), ("a.md", "irrelevant text")]


def test_rerank_chunks_truncates_to_k(monkeypatch):
    class _FakeModel:
        def predict(self, pairs):
            return [float(i) for i in range(len(pairs))]

    monkeypatch.setattr(reranking, "_get_model", lambda: _FakeModel())

    candidates = [(1, "a.md", "a"), (2, "b.md", "b"), (3, "c.md", "c")]
    result = reranking.rerank_chunks("query", candidates, k=1)

    assert result == [("c.md", "c")]


def test_rerank_chunks_uses_the_real_cross_encoder_model():
    candidates = [
        (1, "about.md", "The restaurant is open from 5pm to 10pm on weekdays."),
        (2, "about.md", "Our chef trained in Paris for six years."),
    ]
    result = reranking.rerank_chunks("What time do you open?", candidates, k=1)

    assert result == [("about.md", "The restaurant is open from 5pm to 10pm on weekdays.")]
```

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\pyvenvs\p35backend\Scripts\python.exe -m pytest backend\tests\test_retrieval.py backend\tests\test_reranking.py -v` (from the repo root, or drop the `backend\` prefix if run from `backend/`)
Expected: FAIL — `test_reranking.py`'s tests fail on tuple-shape mismatches (e.g. `not enough values to unpack`), `test_retrieval.py`'s tests fail similarly once `rerank_chunks`/`retrieve_chunks` are still on the old shapes.

- [ ] **Step 3: Implement `app/reranking.py`**

Change the `rerank_chunks` function (leave `_get_model` and everything else in the file
unchanged):

```python
def rerank_chunks(query_text: str, candidates: list[tuple[int, str, str]], k: int) -> list[tuple[str, str]]:
    """Rescore (query, chunk_text) pairs with a cross-encoder and return the
    top-k (source_file, chunk_text) tuples, best first."""
    if not candidates:
        return []
    pairs = [(query_text, chunk_text) for _chunk_id, _source_file, chunk_text in candidates]
    scores = _get_model().predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [(source_file, chunk_text) for (_chunk_id, source_file, chunk_text), _score in ranked[:k]]
```

- [ ] **Step 4: Implement `app/retrieval.py`**

Replace the full contents of `backend/app/retrieval.py`:

```python
import logging

from sqlalchemy import select, text

from app.config import RETRIEVAL_OVERFETCH_K, RETRIEVAL_TOP_K
from app.db import SessionLocal
from app.models import EmbeddingChunk
from app.reranking import rerank_chunks

logger = logging.getLogger("askme")

_RRF_CONSTANT = 60  # standard smoothing constant for reciprocal rank fusion


def _vector_search(session, tenant_id: int, query_embedding: list[float], k: int) -> list[tuple[int, str, str]]:
    stmt = (
        select(EmbeddingChunk.id, EmbeddingChunk.source_file, EmbeddingChunk.chunk_text)
        .where(EmbeddingChunk.tenant_id == tenant_id)
        .order_by(EmbeddingChunk.embedding.cosine_distance(query_embedding))
        .limit(k)
    )
    return [(row.id, row.source_file, row.chunk_text) for row in session.execute(stmt)]


def _fulltext_search(session, tenant_id: int, query_text: str, k: int) -> list[tuple[int, str, str]]:
    rows = session.execute(
        text(
            """
            SELECT id, source_file, chunk_text
            FROM embeddings
            WHERE tenant_id = :tenant_id
              AND chunk_tsv @@ plainto_tsquery('english', :query_text)
            ORDER BY ts_rank(chunk_tsv, plainto_tsquery('english', :query_text)) DESC
            LIMIT :k
            """
        ),
        {"tenant_id": tenant_id, "query_text": query_text, "k": k},
    )
    return [(row.id, row.source_file, row.chunk_text) for row in rows]


def _reciprocal_rank_fusion(ranked_lists: list[list[tuple[int, str, str]]], k: int) -> list[tuple[int, str, str]]:
    """Merge multiple ranked (id, source_file, text) lists into one, scoring each
    id by sum(1 / (_RRF_CONSTANT + rank)) across every list it appears in -- a
    chunk that ranks well in both vector and full-text search outranks one that
    only ranks well in a single list."""
    scores: dict[int, float] = {}
    records: dict[int, tuple[str, str]] = {}
    for ranked in ranked_lists:
        for rank, (chunk_id, source_file, chunk_text) in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (_RRF_CONSTANT + rank)
            records[chunk_id] = (source_file, chunk_text)
    ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [(cid, *records[cid]) for cid in ordered_ids[:k]]


def retrieve_chunks(
    tenant_id: int,
    query_text: str,
    query_embedding: list[float],
    k: int = RETRIEVAL_TOP_K,
    overfetch_k: int = RETRIEVAL_OVERFETCH_K,
) -> list[tuple[str, str]]:
    """Hybrid retrieval: fuse pgvector semantic search with Postgres full-text
    search (reciprocal rank fusion), then rerank the fused candidates with a
    cross-encoder down to the final top-k. Every branch is filtered by
    tenant_id -- the tenant isolation boundary. Returns (source_file, chunk_text)
    tuples so callers can attribute context to a source."""
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
            return [(source_file, chunk_text) for _chunk_id, source_file, chunk_text in fused[:k]]
    finally:
        session.close()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `C:\pyvenvs\p35backend\Scripts\python.exe -m pytest backend\tests\test_retrieval.py backend\tests\test_reranking.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/retrieval.py backend/app/reranking.py backend/tests/test_retrieval.py backend/tests/test_reranking.py
git commit -m "Carry source attribution through hybrid retrieval and reranking"
```

---

## Task 2: Generate-node citations and critique-enforced refusal

**Files:**
- Modify: `backend/app/workflow.py`
- Modify: `backend/app/tracing.py`
- Modify: `backend/tests/test_workflow.py`

**Interfaces:**
- Consumes: `retrieve_chunks` returning `list[tuple[str, str]]` from Task 1.
- Produces: `workflow.REFUSAL_MESSAGE` (module-level string constant) — the exact text
  returned when a retried answer still fails critique. Consumed by tests and available
  for the later observability task to detect declined turns in `chat_trace.log` (a
  decline shows up as `critique_verdicts[-1] == "fail (declined)"` — no schema change
  needed there).

- [ ] **Step 1: Write the failing tests**

In `backend/tests/test_workflow.py`, change `_patch_common`'s mocked `retrieve_chunks`
to return the new tuple shape:

```python
    monkeypatch.setattr(workflow, "retrieve_chunks", lambda tenant_id, query_text, vec, **kw: [("bio.md", "some background chunk")])
```

Replace `test_retries_exactly_once_when_critique_fails` (the retry path now runs a
second critique judgment instead of skipping it, so it needs one more scripted response):

```python
def test_retries_exactly_once_when_critique_fails(monkeypatch):
    # order: classify, generate, critique(fail), generate(retry), critique(pass)
    fake_chat_api = _patch_common(
        monkeypatch, ["hours_location", "first answer", "fail", "second answer", "pass"]
    )

    answer = workflow.run_chat_workflow(tenant_id=1, question="Where did you go to school?")

    assert answer == "second answer"
    assert fake_chat_api.call_count == 5
```

Add a new test right after it:

```python
def test_declines_when_retried_answer_still_fails_critique(monkeypatch):
    # order: classify, generate, critique(fail), generate(retry), critique(fail again)
    fake_chat_api = _patch_common(
        monkeypatch, ["general", "first answer", "fail", "second answer", "fail"]
    )

    answer = workflow.run_chat_workflow(tenant_id=1, question="What do you do?")

    assert answer == workflow.REFUSAL_MESSAGE
    assert fake_chat_api.call_count == 5
```

Add a new test verifying source attribution reaches the generate prompt:

```python
def test_generate_context_includes_source_attribution(monkeypatch):
    fake_chat_api = _patch_common(monkeypatch, ["general", "first answer", "pass"])
    monkeypatch.setattr(
        workflow, "retrieve_chunks",
        lambda tenant_id, query_text, vec, **kw: [("menu.md", "Margherita pizza — $17")],
    )

    workflow.run_chat_workflow(tenant_id=1, question="What pizzas do you have?")

    system_prompt = fake_chat_api.call_kwargs[1]["messages"][0]["content"]
    assert "[Source: menu.md]" in system_prompt
    assert "Margherita pizza — $17" in system_prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\pyvenvs\p35backend\Scripts\python.exe -m pytest backend\tests\test_workflow.py -v`
Expected: FAIL — `test_retries_exactly_once_when_critique_fails` fails because the
current code only makes 4 calls (short-circuits on retry); the new decline/attribution
tests fail because `REFUSAL_MESSAGE` doesn't exist yet and context has no `[Source:
...]` formatting.

- [ ] **Step 3: Implement `app/workflow.py`**

Add a module-level constant near the top of the file, after the imports:

```python
REFUSAL_MESSAGE = "We don't have that information on hand — please ask a staff member directly."
```

Add a helper function above `_generate_node`:

```python
def _format_context(chunks: list[tuple[str, str]]) -> str:
    if not chunks:
        return "(no matching restaurant information found)"
    return "\n\n".join(f"[Source: {source_file}]\n{chunk_text}" for source_file, chunk_text in chunks)
```

Change `_generate_node`'s first line from:
```python
    context = "\n\n".join(state["chunks"]) or "(no matching restaurant information found)"
```
to:
```python
    context = _format_context(state["chunks"])
```

Replace `_critique_node` in full:

```python
def _critique_node(state: ChatState) -> dict:
    already_retried = state["retry_used"]

    try:
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
                        f"CONTEXT: {chr(10).join(chunk_text for _source, chunk_text in state['chunks'])}\n\n"
                        f"ANSWER: {state['answer']}"
                    ),
                },
            ],
        )
    except OpenAIError:
        # We already have a usable answer -- skip enforcement rather than fail the request.
        return {"needs_retry": False}
    verdict = (completion.choices[0].message.content or "pass").strip().lower()
    if verdict.startswith("fail"):
        if already_retried:
            # The retried answer still isn't grounded -- decline rather than
            # return an answer we can't stand behind.
            return {"needs_retry": False, "answer": REFUSAL_MESSAGE}
        return {
            "needs_retry": True,
            "retry_used": True,
            "query": f"{state['question']} (be more specific and grounded)",
        }
    return {"needs_retry": False}
```

- [ ] **Step 4: Update `app/tracing.py`'s critique preview logic**

Change `_preview_for_node`'s signature and critique branch — remove the `prev_retry_used`
parameter entirely (it's no longer needed: the new three-way outcome is fully
determined by `delta`'s keys):

```python
def _preview_for_node(node_name: str, delta: dict) -> str:
    """Compact, truncated one-liner for live console printing."""
    if node_name == "classify":
        return f"category={delta.get('category')}"
    if node_name == "retrieve":
        return f"{len(delta.get('chunks', []))} chunks retrieved"
    if node_name == "generate":
        answer = delta.get("answer") or ""
        snippet = answer[:70] + ("..." if len(answer) > 70 else "")
        return f'answer="{snippet}"'
    if node_name == "critique":
        if delta.get("needs_retry"):
            return "fail (retrying)"
        if "answer" in delta:
            return "fail (declined)"
        return "pass"
    return str(delta)
```

In `trace_turn`, remove the now-unused line `prev_retry_used = final_state.get("retry_used", False)`
and change the call site from `preview = _preview_for_node(node_name, delta, prev_retry_used)`
to `preview = _preview_for_node(node_name, delta)`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `C:\pyvenvs\p35backend\Scripts\python.exe -m pytest backend\tests\test_workflow.py -v`
Expected: all tests PASS, including the 3 new/changed ones.

- [ ] **Step 6: Run the full backend test suite**

Run: `C:\pyvenvs\p35backend\Scripts\python.exe -m pytest backend -q` (or from `backend/`, drop the prefix)
Expected: all tests PASS — this confirms `test_chat_endpoint.py` (which mocks
`tracing.trace_turn` entirely) and any other consumers still work with the new shapes.

- [ ] **Step 7: Commit**

```bash
git add backend/app/workflow.py backend/app/tracing.py backend/tests/test_workflow.py
git commit -m "Enforce citation grounding: source attribution + decline on repeated critique failure"
```
