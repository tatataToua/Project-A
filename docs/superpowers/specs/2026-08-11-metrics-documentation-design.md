# Metrics documentation — design spec

**Date:** 2026-08-11
**Status:** Approved, ready for implementation

## Purpose

The user wants a durable place to record engineering metrics for this project —
performance, cost, quality/correctness, and engineering rigor — primarily so they have
defensible, real numbers to quote as resume/interview bullets ("improved X by Y%") as an
AI engineer. Numbers must be real and reproducible, not estimated, since interviewers
may ask "how did you measure that?"

## Non-goals

- Not building a dashboard, CI metrics pipeline, or automated tracking system.
- Not a general-purpose observability/logging project (that's Phase 5's PostHog work,
  tracked separately in `ROADMAP.md`).
- Not scoped to cover every possible metric — only categories that are both real
  engineering signal and interview-relevant: performance/latency, cost efficiency,
  quality/correctness, engineering rigor.

## Design

### 1. `METRICS.md` (repo root)

Sibling to `ROADMAP.md` and `BUSINESS.md`, following the existing top-level-doc
convention. Three sections:

- **Baseline snapshot** — current numbers as of the date captured, each with a
  methodology note (what was measured, against which LLM provider/model, sample size,
  how to reproduce). Numbers are only useful if their provenance is stated — e.g. today's
  latency numbers are against a local Ollama model (`qwen2.5:7b-instruct` +
  `nomic-embed-text`), not the hosted Gemini free tier, and that distinction must stay
  visible rather than be blurred into a single unlabeled "latency" number.
- **Change ledger** — a table logged over time: `Date | Change | Metric | Before → After
  | Delta | How measured`. A row gets added whenever a change is made that's worth
  quantifying (e.g. Phase 4 Redis caching, Phase 5 model routing/cost governance).
- **Resume bullets** — a short curated list of interview-ready one-liners, derived from
  the ledger, in "action verb + number + mechanism" form. This section is regenerated
  from the ledger, not maintained independently — the ledger is the source of truth.

### 2. `backend/scripts/bench_chat.py`

A committed, reusable version of the ad hoc benchmarking script used to capture today's
baseline. Running it:

- Runs the pytest suite and reports pass count + duration.
- Sends a small fixed set of sample questions through `run_chat_workflow` against the
  live tenant (`two-owls-tavern`), measuring wall-clock latency and token usage per
  query (via the OpenAI-compatible `usage` field on each completion), and reports how
  many LLM calls were made per query (3 = no self-critique retry, 5 = one retry).
- Prints a summary (mean/median/min/max latency, mean tokens, retry rate) in a format
  that can be pasted directly into a new `METRICS.md` ledger row.

This makes every future ledger entry reproducible with one command — the ledger's "how
measured" column can just reference `python backend/scripts/bench_chat.py` plus the git
commit/date, instead of requiring a bespoke one-off measurement each time.

### Today's baseline (captured 2026-08-11, to seed the doc)

- **Tests:** 29/29 passing, 1.92s full suite runtime, 12 backend modules, 607 LOC.
- **Latency** (5 live `/chat` queries, local Ollama `qwen2.5:7b-instruct`): mean 3.77s,
  median 3.96s, range 1.73s–6.72s.
- **Tokens:** mean 1037 total/query (~890 prompt / ~147 completion), 3 LLM calls/query
  (classify → generate → critique), 0/5 queries triggered the self-critique retry.
- **Tenant isolation:** 34 embedded chunks for the one live tenant, retrieval always
  scoped by `tenant_id`.
- **Cost:** $0 today (local Ollama has no per-token cost); if run against the configured
  Gemini free tier instead, cost is also $0 up to the 1500 req/day quota. A real
  cost-per-query dollar figure only becomes meaningful once a paid provider is in the
  loop (relevant once Phase 5's model-routing work lands) — the doc should note this
  rather than fabricate a dollar amount today.

## Testing

No app behavior changes — this is docs + a standalone script. Verification is: the
script runs cleanly end to end and produces numbers consistent with what was captured
manually during design (already confirmed above).
