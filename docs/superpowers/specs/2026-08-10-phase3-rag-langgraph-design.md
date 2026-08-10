# Phase 3: RAG + LangGraph workflow (design)

**Status: approved design, not yet implemented.** Written after a brainstorming session
that reviewed current industry-standard RAG practice (see Research below) and mapped it
onto this project's actual scale and goals. See `ROADMAP.md` for how this phase fits the
overall 5-phase plan, and `docs/superpowers/specs/2026-08-10-ask-me-saas-design.md` for
the tenant-aware architecture this design builds on.

## Why this exists

Today, `/chat` (`backend/app/main.py`) builds its system prompt by reading `bio.md`
whole and pasting it into the prompt on every request — no retrieval, no chunking, only
one file used (`projects.md` and `resume.md` sit unread). This doesn't scale past small
content, wastes tokens on irrelevant text, and doesn't demonstrate the retrieval/
orchestration skills this project exists to practice (per `ROADMAP.md`'s framing as a
learning vehicle through the AI Engineer pyramid). This design replaces that with real
chunked/embedded retrieval plus a small LangGraph orchestration loop.

## Scope decision

This design covers **retrieval (RAG) and the LangGraph classify→retrieve→generate→
self-critique loop only**. The MCP tool (`fetch_github_activity`) originally scoped as
part of Phase 3 in `ROADMAP.md` is deliberately deferred to its own future design pass —
there's no concrete need for it yet, and folding it in now would be building against a
requirement that doesn't exist.

## Research grounding

Current (2026) industry practice for production RAG was reviewed before this design:

- Chunking is the highest-leverage layer — most RAG failures trace to bad ingestion/
  chunking, not the LLM. Structure-aware chunking (splitting on semantic/structural
  boundaries) is preferred over fixed-size splitting.
- Full production RAG stacks add hybrid search, reranking, query rewriting, and semantic
  caching — but that stack is built for enterprise-scale corpora. This project's content
  (three short markdown files) doesn't warrant that complexity yet; the "minimal correct
  RAG" (chunk → embed → cosine similarity retrieval) proves the same core pattern without
  building machinery that has nothing to prove itself against at this scale.
- LangGraph is justified when a workflow needs loops, conditional routing, or multi-step
  state — not for genuinely linear problems. Here it's adopted deliberately for the
  learning/portfolio goal (this is one of the skills `ROADMAP.md` calls out as the point
  of this phase), even though a plain bio-Q&A bot doesn't strictly require it.

## Architecture

### Data model

- **`tenants`** — `id`, `slug`, `name`, `created_at`, `status`. Rows created manually (no
  signup flow yet, per the SaaS spec). Only one row exists initially: the owner's own
  instance.
- **`embeddings`** — `id`, `tenant_id` (FK), `source_file` (e.g. `"bio.md"`),
  `chunk_text`, `chunk_index`, `embedding vector(...)`. Every retrieval query filters by
  `tenant_id` — per the SaaS spec, cross-tenant leakage is treated as a correctness bug,
  not a cosmetic one, and gets explicit test coverage (see Testing).
- Content stays as files (`docs/content/tenants/<slug>/{bio,projects,resume}.md`); the
  `embeddings` table is a derived index over them, not a replacement.

### Ingestion & chunking

A new ingestion module/script (e.g. `backend/app/ingest.py`) that:

1. Reads each tenant's `bio.md`, `projects.md`, `resume.md`.
2. Splits each file by markdown structural boundaries (`##`/`###` headings) so each
   chunk is a self-contained section (one job, one project, one skill area) rather than
   an arbitrary fixed-size slice.
3. Embeds each chunk via Gemini's embedding model (`gemini-embedding-001`, default —
   same provider already used for generation, no new API key/billing to set up). New
   config constant `GEMINI_EMBEDDING_MODEL` added to `config.py`, following the existing
   pattern for provider/model constants.
4. Stores `(tenant_id, source_file, chunk_text, chunk_index, embedding)` rows in
   `embeddings`.

Run manually (e.g. `python -m app.ingest <tenant_slug>`) whenever content changes — no
file-watcher or auto-reingest yet; content changes rarely and the operator is the only
user of this tool during the concierge stage.

### Retrieval

Plain vector similarity search, no hybrid search or reranking:

1. Embed the incoming user question with the same Gemini embedding model, so it lands in
   the same vector space as stored chunks.
2. Query `embeddings` for the top-k (e.g. top 5) chunks by cosine similarity, filtered by
   `tenant_id`.
3. Concatenate retrieved chunks into the prompt, replacing today's whole-file dump.

### LangGraph workflow

Replaces the current single `client.chat.completions.create()` call in `/chat` with a
small stateful graph:

```
question
  |
  v
classify --> "about background" --+
  |                                |
  +--> "about a project" ---------+--> retrieve (tenant-scoped) --> generate --> self-critique
                                   |                                                |
  "general/other" ------------------                                       pass? --+--> respond
                                                                               |
                                                                        fail (max 1 retry)
                                                                               |
                                                                               v
                                                                  retrieve (revised query) --> generate --> respond
```

- **classify**: one LLM call; routes the question into a category to bias the retrieval
  query (e.g. "about a project" favors `projects.md` chunks).
- **retrieve**: the vector search described above.
- **generate**: the existing Gemini call, now given retrieved chunks instead of the
  whole file.
- **self-critique**: one more LLM call checking whether the answer is grounded in the
  retrieved context and addresses the question. On failure, retries retrieval+generation
  once with a revised query, then answers regardless — capped at one retry, no infinite
  loop.

The tool-call branch from the original `ROADMAP.md` sketch (`fetch_github_activity`) is
intentionally not part of this graph; see Scope decision above.

### API & error handling

- `/chat` becomes `/chat/{tenant_slug}` (per the SaaS spec). Only one tenant slug
  resolves today, but the route shape is correct from the start.
- `frontend/src/ChatWidget.jsx` gains a `tenant` identifier it includes in the request
  URL — a small change since it's a single instance today.
- Unknown tenant slug → 404.
- Tenant exists but has no ingested content (ingest script never run) → the assistant
  explicitly says so rather than falling back to whole-file behavior or hallucinating.
- Gemini call failures → unchanged; existing 502 handling in `main.py` stays as-is.

## Dependencies

- `langgraph` added to `backend/requirements.txt` for the orchestration graph.
- pgvector is already provisioned via `docker-compose` (per `ROADMAP.md` Phase 1) but
  unused until now — `db.py`'s existing SQLAlchemy engine is extended with the
  `embeddings`/`tenants` tables and a pgvector column type (e.g. via `pgvector-python`'s
  SQLAlchemy integration).

## Testing

- **Tenant isolation test**: seed two tenants with distinct content; assert tenant A's
  retrieval never returns tenant B's chunks. Flagged by the SaaS spec as the one piece
  that must be verified before a second real tenant exists — cheap to verify now while
  the schema is new.
- **Chunking sanity test**: given a sample markdown file, assert it splits into the
  expected number of heading-bounded chunks.
- **Retrieval relevance spot-check**: a handful of known question → expected-chunk pairs
  (e.g. "where did you go to school" should retrieve the education chunk, not the
  projects chunk). Informal for now — the full DeepEval golden-set suite is Phase 5.
- **Self-critique loop test**: force a low-quality first answer (mocked) and assert the
  retry path fires exactly once, not infinitely.

## Non-goals (for this slice)

- The MCP tool (`fetch_github_activity`) — deferred to its own design.
- Hybrid search, reranking, query rewriting, semantic caching — deferred until content
  scale or measured retrieval quality actually motivates them.
- Self-serve tenant signup, billing, admin dashboard — unchanged non-goals from the SaaS
  spec.
- Auto-reingestion on file change — ingestion stays a manual script for now.

## Open questions

- Chunk size/count in practice depends on how `bio.md`/`projects.md`/`resume.md` are
  actually structured today — the heading-based chunker's behavior should be sanity-
  checked against the real files during implementation, not just assumed.
- Top-k for retrieval (5, suggested above) is a starting guess, not benchmarked against
  this content — worth revisiting once the relevance spot-check test exists.
- If the SaaS direction in `BUSINESS.md` doesn't pan out, the tenant_id scoping built
  here was built ahead of validated need — accepted trade-off per the earlier design
  session, not a mistake to fix reactively.
