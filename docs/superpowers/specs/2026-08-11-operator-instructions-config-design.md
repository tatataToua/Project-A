# Operator-tunable custom instructions — design spec

**Date:** 2026-08-11
**Status:** Approved, ready for implementation

## Purpose

The chat pipeline's tone, verbosity, and topic scope are currently fixed by the
hardcoded system prompt in `_generate_node` (`backend/app/workflow.py`). The operator
(app owner, not an end user) wants a fast way to nudge that behavior — e.g. "keep
answers shorter" or "never answer questions unrelated to the restaurant" — without
editing Python or redeploying. This adds a single local text file the operator edits
directly; its contents are appended to the generate-node system prompt on every chat
request.

This is deliberately not a per-tenant persona config (see
`2026-08-11-restaurant-classify-persona-design.md`, whose non-goals section already
covers why that's speculative with one tenant live). It's a global, operator-facing
tuning knob, analogous in spirit to `backend/.env` — a local file that changes runtime
behavior, gitignored, not shipped as tenant content.

## Design

### 1. Instructions file

`backend/instructions.txt` — plain text, gitignored. Read fresh on every `/chat`
request (no caching): it's one small file read, negligible next to the LLM calls
already made per request, and skipping a cache avoids mtime-check bookkeeping for no
real benefit. Editing and saving the file takes effect on the very next chat message,
no restart needed.

A committed `backend/instructions.txt.example` documents the format with sample
content, e.g.:

```
Keep answers to 2-3 sentences.
Only answer questions about Two Owls Tavern; politely decline anything else.
```

`.gitignore` gains a `backend/instructions.txt` entry alongside the existing `.env`
line.

### 2. Reading it (`backend/app/instructions.py`, new module)

One function:

```python
def get_custom_instructions() -> str:
    ...
```

Reads `config.INSTRUCTIONS_FILE`, returns its stripped contents, or `""` if the file
doesn't exist or is empty/whitespace-only. No exceptions escape this function for the
"file missing" case — that's the expected default state, not an error.

`config.py` gains one constant, following the existing `CONTENT_DIR` pattern:

```python
INSTRUCTIONS_FILE = REPO_ROOT / "backend" / "instructions.txt"
```

### 3. Wiring into the generate node (`backend/app/workflow.py`)

`_generate_node` calls `get_custom_instructions()`. If non-empty, it's appended to the
existing system prompt as a separately delimited block, placed **after** the core
prompt (grounding-in-context instruction, first-person-plural voice, allergy-safety
guardrail):

```python
system_prompt = (
    "You are the AI assistant for a restaurant, ..."
    f"--- CONTEXT ---\n{context}"
)
custom_instructions = get_custom_instructions()
if custom_instructions:
    system_prompt += (
        f"\n\n--- ADDITIONAL OPERATOR INSTRUCTIONS ---\n{custom_instructions}"
    )
```

Additive placement is intentional: the operator can retune tone/scope, but the base
grounding and allergy-safety instructions stay in the prompt regardless of what the
operator writes. `_classify_node`, `_retrieve_node`, and `_critique_node` are
untouched — topic-scope instructions like "only answer about the restaurant" are
enforced by the LLM honoring the generate-node prompt, not by new branching logic.

### 4. Tests (`backend/tests/test_workflow.py`)

Add a test that patches `get_custom_instructions` to return a non-empty string and
asserts it appears in the system prompt passed to the mocked LLM client, plus a test
that an empty/missing file produces no such block (baseline behavior unchanged from
existing tests). Existing classify/retrieve/critique tests need no changes since only
`_generate_node`'s prompt construction changes.

## Testing

Run the backend test suite (`.venv\Scripts\python.exe -m pytest` from `backend/`)
after adding the new tests; all tests should pass. Manual check: create
`backend/instructions.txt` with a short instruction (e.g. "Answer in one sentence
only."), ask a question via the running app, confirm the response honors it; then
delete/empty the file and confirm behavior reverts without a restart.
