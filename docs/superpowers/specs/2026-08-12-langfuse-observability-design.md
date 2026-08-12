# Langfuse observability + golden eval — design spec

**Date:** 2026-08-12
**Status:** Approved, ready for implementation

## Purpose

Two manual workflows currently stand in for real observability and quality tracking:

1. **Trace inspection.** `backend/app/tracing.py`'s `trace_turn()` hand-builds a
   per-node timing/token/critique record for every real `/chat/{tenant_slug}` call and
   appends it as JSONL to `backend/logs/chat_trace.log`. Seeing anything useful means
   running `python -m app.trace_chat --watch`, `--stats`, or `--export-csv` and reading
   the output yourself.
2. **Quality checks.** There is no golden Q&A regression set today — quality is
   verified by manual spot-checking (the kind of ad hoc check called out as backlog in
   `2026-08-11-operator-instructions-config-design.md`'s non-goals).

This integrates [Langfuse](https://langfuse.com) (cloud, free tier) to replace both:
traces get sent automatically and are browsable/searchable/filterable in Langfuse's
hosted UI instead of a local log file, and a golden Q&A set becomes a Langfuse Dataset
that a runner script executes as scored Experiments, so quality is trackable over time
rather than spot-checked.

## Non-goals

- **CI gate.** The eval runner is a local/manual script (like `bench_chat.py` today),
  not wired into GitHub Actions. Worth adding once the golden set has proven itself
  useful; premature to gate merges on it now.
- **LLM-as-judge scoring.** v1 scores golden-eval answers with a deterministic
  keyword-containment check, not an LLM judge. An LLM-judge score is a natural
  follow-up (mirrors the existing critique-node pattern) but adds cost, latency, and
  judge-prompt design to get right — not needed to get regression tracking working.
- **Self-hosting Langfuse.** Cloud free tier only; no new docker-compose service.
- **Removing/replacing the critique node's retry logic.** Unrelated — this only adds
  observability and scoring around the existing workflow, not changes to it.

## Design

### 1. Instrumentation entry point (`backend/app/llm.py`)

Swap the raw OpenAI client for Langfuse's drop-in wrapper:

```python
from langfuse.openai import OpenAI

from app.config import GEMINI_API_KEY, GEMINI_BASE_URL

client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)
```

Same constructor signature, same `client.chat.completions.create` /
`client.embeddings.create` call sites everywhere (`workflow.py`, `embeddings.py`,
`trace_chat.py`, the new eval runner) — every call becomes an auto-captured Generation
with tokens and cost, nested under whatever Langfuse span is active when it runs. This
is the one change that makes every downstream call site "just work" with no per-call
edits.

### 2. Per-request tracing (`backend/app/tracing.py`)

Remove:
- `_usage_buffer`, `instrument_client()` — token capture is now automatic via `llm.py`.
- `LOG_PATH` and the `open(...).write(...)` JSONL append at the end of `trace_turn()`.

Keep `trace_turn(tenant_id, question, on_node=None)`'s signature and its
`_graph.stream(state, stream_mode="updates")` loop — still the mechanism for live
per-node console output in the REPL (see §3) and for deriving `first_pass`/`retried`
for the return value. Change the call to pass Langfuse's LangGraph callback handler:

```python
from langfuse import get_client
from langfuse.langchain import CallbackHandler

langfuse = get_client()

def trace_turn(tenant_id, question, on_node=None, tenant_slug=None, user_email=None):
    handler = CallbackHandler()
    with langfuse.start_as_current_span(name="chat_turn") as span:
        span.update_trace(
            tags=[tenant_slug] if tenant_slug else None,
            user_id=user_email,
            input={"question": question},
        )
        for update in _graph.stream(state, config={"callbacks": [handler]}, stream_mode="updates"):
            ...  # unchanged per-node bookkeeping for on_node/first_pass/retried
        span.update_trace(output={"answer": answer})
        langfuse.score_current_trace(name="critique_pass", value=first_pass, data_type="BOOLEAN")
    return answer, record
```

`record` (the return value) drops `events`/`node_latency_s`/`node_tokens` — those live
in Langfuse now — and keeps just what `main.py`/the REPL still need: `answer`,
`total_latency_s`, `first_pass`, `retried`.

`main.py`'s call site needs `tenant_slug` and the logged-in user's email threaded
through (both already available where `trace_turn` is called — `req` has the tenant
slug from the route, and `require_user` puts the session user on the request). No
change to the endpoint's response shape.

### 3. `trace_chat.py`

Remove `--watch`, `--stats`, `--export-csv`, their handler functions
(`_watch`/`_print_stats`/`_export_csv`), and `CSV_FIELDNAMES` — Langfuse's UI covers
tailing, aggregate stats, and export.

Keep the interactive REPL path (`_run_turn`, `_format_summary`, `_format_node_line`,
`_preview_for_node`-driven live printing). After each turn, print the Langfuse trace
URL (`langfuse.get_trace_url()` or equivalent from the SDK) so a turn can be clicked
straight into. `main()` loses the `--watch`/`--stats`/`--export-csv` argparse options
but keeps the REPL default and the `tenant_slug` positional arg.

### 4. Golden eval (new)

**Golden set storage:** `backend/evals/golden_sets/<tenant_slug>.json` — a plain JSON
array, source of truth, reviewed like any other code change:

```json
[
  {
    "question": "What time do you open on Saturdays?",
    "category": "hours_location",
    "must_mention": ["Saturday", "hours-or-time-string"]
  }
]
```

Start with ~10-15 hand-authored items for `two-owls-tavern` spanning all four
classify-node categories (menu, hours_location, policies, general).

**Sync script:** `backend/scripts/sync_golden_dataset.py <tenant_slug>` — reads the
JSON file, upserts each item into a Langfuse Dataset named `<tenant_slug>-golden`
(create the dataset if missing; upsert items keyed by `question` so re-running after
an edit is idempotent rather than accumulating duplicates).

**Eval runner:** `backend/scripts/run_eval.py <tenant_slug> [--run-name NAME]` —
fetches the dataset's items from Langfuse, and for each one:
1. Runs it through `run_chat_workflow(tenant_id, question)` inside the dataset item's
   `run(run_name=...)` context (Langfuse SDK links the resulting trace to that dataset
   item + run automatically).
2. Scores it: case-insensitive substring check that every string in `must_mention`
   appears in the answer; push via `langfuse.score()` on that run's trace.
3. Prints a local summary (pass rate, list of failing questions) to the console —
   default `run_name` is a timestamp if not given.

This is a standalone operator script in the same spirit as `bench_chat.py` — not a
pytest test, run by hand: `python -m backend.scripts.run_eval two-owls-tavern`.

### 5. Config

`config.py` gains, following the existing pattern:

```python
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
```

`.env.example` gains the three keys with a comment pointing at
https://cloud.langfuse.com for signup/key generation. CLAUDE.md's env var table gains
a row for each. `requirements.txt` gains `langfuse`.

### 6. Error handling

Langfuse's SDK batches and flushes observations on a background thread — a Langfuse
outage or misconfigured/missing keys must not fail a `/chat` request. This needs an
explicit manual check during implementation: start the app with deliberately invalid
`LANGFUSE_*` values and confirm `/chat` still returns a normal answer (Langfuse client
logs/no-ops rather than raising).

### 7. Tests

`test_workflow.py` mocks `_client.chat.completions.create` directly today. Since
`llm.py`'s `client` becomes a `langfuse.openai.OpenAI` instance, confirm that
`.chat.completions.create` is still the right attribute to patch (it should be — the
wrapper preserves the same object shape) by running the existing suite unchanged
first, before making other changes, as a compatibility check.

Add an autouse fixture in `backend/tests/conftest.py` that ensures Langfuse doesn't
attempt real network calls or block test teardown during the suite — e.g. leaving
`LANGFUSE_*` env vars unset in the test environment (so the client no-ops) and
confirming no test hangs on flush; make this explicit rather than assumed.

No new tests for `run_eval.py`/`sync_golden_dataset.py` beyond manual verification —
they're operator scripts in the same category as `bench_chat.py`, which also has no
test coverage today.

## Testing

1. Run `.venv\Scripts\python.exe -m pytest` from `backend/` after each stage — the
   `llm.py` swap first (confirm existing mocks still pass), then the `tracing.py`
   rework.
2. Manual: start the app, ask a question via the browser widget, confirm a trace
   appears in the Langfuse dashboard with nested classify/retrieve/generate/critique
   spans, correct token counts, and a `critique_pass` score.
3. Manual: run `python -m app.trace_chat two-owls-tavern`, ask a question, confirm
   live per-node console output still works and a trace URL is printed.
4. Manual: `python -m backend.scripts.sync_golden_dataset two-owls-tavern`, confirm
   the dataset appears in Langfuse with the right item count; then
   `python -m backend.scripts.run_eval two-owls-tavern`, confirm a pass-rate summary
   prints and an Experiment run with scores appears in Langfuse.
5. Manual: temporarily set an invalid `LANGFUSE_SECRET_KEY`, confirm `/chat` still
   returns a normal answer (error-handling check from §6).
