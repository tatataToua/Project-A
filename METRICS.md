# Metrics

A running record of real, reproducible numbers for this project — performance, cost,
quality, and engineering rigor — kept so that improvements have defensible before/after
evidence instead of vague claims. See `docs/superpowers/specs/2026-08-11-metrics-documentation-design.md`
for the design rationale.

**How to reproduce any number below:** run `backend/scripts/bench_chat.py` (from
`backend/`, with the venv active, Postgres up, and the `two-owls-tavern` tenant already
ingested). It runs the test suite and a fixed set of sample `/chat` queries, then prints
a summary block formatted to paste straight into the ledger below.

```bash
cd backend
.venv\Scripts\python.exe scripts\bench_chat.py
```

## Baseline snapshot (2026-08-11)

| Metric | Value | Notes |
|---|---|---|
| Tests | 29/29 passing, 1.94s | 12 backend modules, 607 LOC (`app/`) |
| Chat latency | mean 1.61s, median 1.63s, range 1.41s–1.78s (n=5) | Against **local Ollama** (`qwen2.5:7b-instruct` + `nomic-embed-text`), not hosted Gemini — see Provider note below |
| Tokens per query | mean 1024, range 868–1160 | 3 LLM calls/query (classify → generate → critique) |
| Self-critique retry rate | 0/5 in this sample | Retry path exists (`workflow.py`) but wasn't exercised by these 5 questions |
| Tenant isolation | 34 embedded chunks, 1 tenant (`two-owls-tavern`) | Every retrieval query filtered by `tenant_id` (`retrieval.py`) |
| Cost per query | $0 today | Local Ollama has no per-token cost; the app is also verified against the hosted Gemini free tier (1500 req/day, also $0 in-quota). A real $/query figure needs a paid provider in the loop — revisit once Phase 5 model routing lands. |

**Provider note:** this backend is deliberately provider-swappable (`app/llm.py`) — the
same code path has been verified against both Gemini's OpenAI-compatible endpoint and a
local Ollama server. Baseline numbers here are provider-specific; always check
`GEMINI_BASE_URL`/`GEMINI_MODEL` in `.env` before comparing two ledger rows, or you'll be
comparing apples to oranges (e.g. local 7B model latency vs. hosted API latency).

## Change ledger

| Date | Change | Metric | Before → After | Delta | How measured |
|---|---|---|---|---|---|
| 2026-08-11 | Initial baseline captured | — | — | — | `backend/scripts/bench_chat.py` |

*(Add a row here each time a change is worth quantifying — e.g. Phase 4's Redis
caching, Phase 5's model routing / cost governance.)*

## Resume bullets

*(Derived from the ledger above — regenerate this list as new rows land, don't maintain
it independently.)*

- Built a tenant-isolated RAG pipeline (FastAPI + pgvector + LangGraph) with 29 passing
  tests covering retrieval isolation, auth, rate limiting, and workflow retry logic.
- Designed the LLM client layer to be provider-swappable with zero call-site changes —
  verified working against both a hosted API (Gemini) and a local model (Ollama
  `qwen2.5:7b-instruct`).
- Implemented a self-critique retry loop in the chat workflow (LangGraph) so answers
  that fail a groundedness check get one automatic re-retrieval-and-regenerate pass
  before reaching the user.

*(These will get more concrete as ledger rows accumulate — e.g. "cut latency 40% via
caching" once that change is made and measured.)*
