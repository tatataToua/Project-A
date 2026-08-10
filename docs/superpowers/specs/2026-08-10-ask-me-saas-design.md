# Ask Me: multi-tenant SaaS design (working hypothesis)

**Status: not validated, not committed.** This spec documents a direction decided in one
brainstorming session, at the explicit request to have something concrete written down
because the end-state was "still unsure." It should make future decisions easier to
reason about, not harder to change. If real customer conversations (see `BUSINESS.md`)
contradict any assumption here, update this doc rather than forcing the build to match it.

## Why this exists

`ROADMAP.md` describes "Ask Me" as a single-tenant personal AI assistant (answers
questions about one person, grounded in their own bio/projects/resume), built in 5
phases as a learning vehicle for AI engineering skills. That's still true and still the
near-term build. This spec adds a second, simultaneous goal: architect it so the same
system can become a multi-tenant SaaS product later, without a rewrite — because the
project owner wants both the skill-building AND a shot at a real business, and doesn't
want to choose between them prematurely.

See `BUSINESS.md` for the business rationale (niche, pricing, go-to-market) behind why
multi-tenancy specifically, and for the niche/pricing assumptions this design depends on.

## Scope decision: what "done" means for this slice

Per the brainstorming session:
- The personal instance ("Ask Me about Toua") becomes **tenant #1** inside the
  multi-tenant system, not a separate codebase — this repo evolves in place.
- Go-to-market is **concierge-first**: the owner manually finds and onboards the first
  ~10-20 customers. This means **no self-serve signup, billing, or admin dashboard UI
  is in scope yet** — those are explicit non-goals until the sales motion is proven
  repeatable (see `BUSINESS.md`).
- The existing 5-phase roadmap (`ROADMAP.md`) stays the implementation sequence; this
  spec changes *how* Phase 3 onward gets built (tenant-aware from the start) rather than
  replacing the plan.

## Architecture

Foundation is unchanged from `ROADMAP.md`: FastAPI backend, Postgres + pgvector,
React/Vite embeddable widget. The change is making retrieval and content tenant-scoped
instead of assuming a single global bio.

### Data model

- **`tenants`** — `id`, `slug`, `name`, `created_at`, `status`. Rows are created
  manually (via an admin script or a single shared-secret-protected endpoint) — there
  is no public signup flow in this phase.
- **Content** stays as markdown files, matching the existing `docs/content/*.md`
  pattern, namespaced per tenant: `docs/content/tenants/<slug>/{bio,services,faq}.md`.
  Files over a database-backed CMS because onboarding is manual anyway at this stage —
  a database-editable content model is deferred until self-serve onboarding exists.
- **`embeddings`** (new, Phase 3) — chunked content + vector + `tenant_id`. Every
  retrieval query must filter by `tenant_id`. This is the single highest-risk correctness
  bug in the system: a cross-tenant leak (tenant A's answer surfacing tenant B's private
  content) is a trust-breaking failure, not a cosmetic one, so it gets explicit test
  coverage (see Testing below), not just implicit correctness from the query shape.
- **`conversations` / `messages`** (new, pulled forward from Phase 5) — logged per
  tenant. This is data Phase 5's eval suite needs anyway; tagging it by tenant from the
  start means it can also become a real product feature later ("see what your visitors
  are asking") at no extra cost.

### API & widget

- `/chat` becomes tenant-scoped: `/chat/{tenant_slug}`. The handler builds the system
  prompt and retrieves context using only that tenant's embedded content.
- The embeddable widget (`frontend/src/ChatWidget.jsx`) takes a `tenant` identifier
  (e.g. a `data-tenant` attribute on the embed snippet). Each customer's snippet is
  generated and handed to them manually — no self-serve embed generator yet.

### Provisioning

A minimal internal tool — a script or one admin-only endpoint — that:
1. Creates a `tenants` row.
2. Runs the ingest/chunk/embed pipeline (Phase 3) against that tenant's content folder.

Deliberately not a dashboard. The owner is the only operator during the concierge phase;
building admin UI now would be scope that doesn't serve the current goal.

## Error handling

- Unknown tenant slug → 404.
- Tenant exists but has no ingested content yet → the assistant says so explicitly
  rather than answering from no context (avoids hallucination dressed as an answer).
- LLM/API call failures → existing frontend fallback behavior in `ChatWidget.jsx`
  already handles this; no change needed.

## Testing

- Phase 5's planned DeepEval golden-set testing gets one golden Q&A set per tenant,
  starting with tenant #1 (the owner's own instance).
- New test, specific to multi-tenancy: an isolation test that asserts tenant A's
  retrieval never surfaces tenant B's content, run against at least two seeded tenants.
  This is the one piece of this design that must be verified before onboarding a second
  real customer, since it's the failure mode that would break trust immediately.

## Non-goals (for this slice)

Self-serve signup, billing/payments (Stripe), admin dashboard UI, public marketing site.
Revisit once the concierge sales motion produces a repeatable pattern (per `BUSINESS.md`'s
"~20 paying customers" validation bar) — building these earlier would be scope the
current stage doesn't need yet.

## Open questions

Same as `BUSINESS.md`'s open questions — this design is downstream of the business
niche/pricing hypothesis, so if that changes (e.g. urgency turns out to be too weak in
real conversations), this architecture should be re-examined too, particularly whether
markdown-file content storage still makes sense at a different scale or niche.
