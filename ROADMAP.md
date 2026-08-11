# Roadmap

"Ask Me" is built in 5 phases, one per level of the AI Engineer pyramid from
[How to Become an AI Engineer FAST (2026)](https://www.youtube.com/watch?v=aAItDrJ8-rE).
Each phase is a real increment to the same app — by the end this is one deployed,
instrumented AI product, not five disconnected demos. See `docs/superpowers/specs/`
(or the original design conversation) for the full rationale behind each tech choice.

**Working hypothesis, as of 2026-08-11:** this is also being architected as a potential
multi-tenant SaaS product, not just a personal instance. Not validated or committed —
see `BUSINESS.md` for the business rationale and
`docs/superpowers/specs/2026-08-10-ask-me-saas-design.md` for how that shaped Phase 3
(tenant-aware from the start, isolation enforced at every retrieval query). The current
demo tenant is **Two Owls Tavern** (a fictional restaurant,
`docs/content/tenants/two-owls-tavern/`) rather than a personal bio — chosen to prove
the pipeline works for non-personal, business-shaped content (hours, menu, FAQs), not
just a single person's background. This validates the technical assumption; it is not
business validation (no real customer, no paying tenant yet — see `BUSINESS.md`'s
"first real validation" definition).

- [x] **Phase 1 — SALT Foundation.** Repo scaffold, FastAPI skeleton (`/health`), Postgres
      + pgvector via Docker Compose, Git. Condensed since you're an experienced engineer —
      this is a checklist, not a teaching module.
- [x] **Phase 2 — Controlled Intelligence (first slice).** `/chat` endpoint called an LLM
      directly with a persona system prompt built from a single flat content file. No
      retrieval yet — proved basic API integration + prompt engineering. Using
      **Gemini 2.5 Flash** (free tier, no credit card, 1500 req/day) via Google's
      OpenAI-compatible endpoint instead of paid OpenAI — same `openai` SDK code path,
      just a different `base_url`/key/model. Fully superseded by Phase 3 (the flat
      content file and `build_system_prompt()` no longer exist).
- [x] **Phase 3 — Intelligent Systems.** Content is chunked on markdown heading
      boundaries (`chunking.py`), embedded (`embeddings.py`), and stored per-tenant in
      pgvector (`models.py`) via an offline ingest step (`ingest.py`) — every row and
      every retrieval query is scoped by `tenant_id`, treated as a correctness
      requirement per the SaaS design spec, not a later add-on. A LangGraph
      classify → retrieve → generate → self-critique workflow (`workflow.py`, one retry
      max) orchestrates each request, invoked from `POST /chat/{tenant_slug}`. LLM
      provider is swappable via one shared client (`app/llm.py`) — verified working
      against both Gemini and a local Ollama server.
  - **Not built, carried forward:** the MCP tool (e.g. `fetch_github_activity`) from the
    original Phase 3 scope was explicitly deferred — see the design spec's Scope
    Decision / Non-goals. Revisit if/when a tenant's content benefits from live
    structured data instead of only static markdown.
- [ ] **Phase 4 — Scaling.**
  - Dockerize the FastAPI backend (and optionally serve the built widget from it).
  - Deploy to AWS (ECS Fargate or App Runner) with a real public URL.
  - Embed the widget on your actual personal site.
  - Add Redis caching for repeated/common questions to cut latency and OpenAI cost.
- [ ] **Phase 5 — Strategic AI Operations (LLMOps).**
  - DeepEval test suite against a golden set of Q&A about you — catches hallucinations
    and regressions as the prompt/retrieval logic changes.
  - PostHog (or Amplitude) analytics: what visitors ask, where they drop off.
  - Cost governance / model routing: a cheap model (e.g. `gpt-4o-mini`) for simple
    FAQ-style questions, escalate to a stronger model for nuanced ones, log cost per
    conversation.

## How to pick this back up

Each future session should target exactly one unchecked phase above. Check the box
when a phase's checklist is genuinely done (not just started) and update the "First
slice" caveats in this file as they get filled out.
