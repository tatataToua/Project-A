# Roadmap

"Ask Me" is built in 5 phases, one per level of the AI Engineer pyramid from
[How to Become an AI Engineer FAST (2026)](https://www.youtube.com/watch?v=aAItDrJ8-rE).
Each phase is a real increment to the same app — by the end this is one deployed,
instrumented AI product, not five disconnected demos. See `docs/superpowers/specs/`
(or the original design conversation) for the full rationale behind each tech choice.

**Working hypothesis, as of 2026-08-10:** this is also being architected as a potential
multi-tenant SaaS product (not just a personal instance), with the owner's own instance
as tenant #1. Not validated or committed — see `BUSINESS.md` for the business rationale
and `docs/superpowers/specs/2026-08-10-ask-me-saas-design.md` for how that changes the
technical plan below (mainly: Phase 3 becomes tenant-aware from the start).

- [x] **Phase 1 — SALT Foundation.** Repo scaffold, FastAPI skeleton (`/health`), Postgres
      + pgvector via Docker Compose, Git. Condensed since you're an experienced engineer —
      this is a checklist, not a teaching module.
- [x] **Phase 2 — Controlled Intelligence (first slice).** `/chat` endpoint calls an LLM
      directly with a persona system prompt sourced from `docs/content/bio.md`. No
      retrieval yet — proves basic API integration + prompt engineering. Using
      **Gemini 2.5 Flash** (free tier, no credit card, 1500 req/day) via Google's
      OpenAI-compatible endpoint instead of paid OpenAI — same `openai` SDK code path,
      just a different `base_url`/key/model, so switching to real OpenAI later is a
      one-line change in `backend/app/main.py`.
- [ ] **Phase 3 — Intelligent Systems.**
  - Chunk + embed content into Postgres via pgvector, tagged by `tenant_id` from the
    start (content lives at `docs/content/tenants/<slug>/*.md`; the owner's own content
    becomes tenant #1). See the SaaS design spec for the data model and why isolation
    between tenants is treated as a correctness requirement, not a later add-on.
  - Real RAG: retrieve relevant chunks for a question (scoped to one tenant), insert
    into the prompt, generate.
  - A LangGraph multi-step workflow: classify the question (about background / a specific
    project / general advice) → retrieve → optionally call a tool → generate → self-critique.
  - One MCP tool, e.g. `fetch_github_activity`, so the assistant can pull live, structured
    data about your real projects instead of only static text from `projects.md`.
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
