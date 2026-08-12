# Operator-Tunable Custom Instructions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the operator drop free-text tuning instructions (tone, verbosity, topic
scope) into a local `backend/instructions.txt` file that gets appended to the chat
pipeline's generate-node system prompt on every request, with no restart required.

**Architecture:** A new one-function module (`app/instructions.py`) reads the file
fresh on each call and returns its stripped contents (or `""` if missing/empty).
`workflow.py`'s `_generate_node` calls it and, if non-empty, appends the text to the
existing system prompt as a clearly delimited block *after* the core prompt and
allergy-safety guardrail — additive, never a replacement.

**Tech Stack:** Python (FastAPI backend), pytest + `monkeypatch` for testing (matches
existing `backend/tests/test_workflow.py` patterns — no new dependencies).

## Global Constraints

- File path constant lives in `backend/app/config.py`, following the existing
  `CONTENT_DIR = REPO_ROOT / "docs" / "content"` pattern (spec section 2).
- `backend/instructions.txt` itself is gitignored; a `backend/instructions.txt.example`
  is committed to document the format (spec section 1).
- No caching of file contents — read fresh on every call (spec section 1).
- The custom-instructions block must be appended *after* the core system prompt, never
  replacing or reordering it (spec section 3).
- Only `_generate_node` changes in `workflow.py`; `_classify_node`, `_retrieve_node`,
  `_critique_node` are untouched (spec section 3).

---

### Task 1: Custom-instructions reader (`app/instructions.py`)

**Files:**
- Modify: `backend/app/config.py` (add `INSTRUCTIONS_FILE` constant)
- Create: `backend/app/instructions.py`
- Create: `backend/instructions.txt.example`
- Modify: `.gitignore` (add `backend/instructions.txt`)
- Test: `backend/tests/test_instructions.py`

**Interfaces:**
- Produces: `app.instructions.get_custom_instructions() -> str` — returns the
  stripped file contents, or `""` if the file is missing or contains only
  whitespace. Consumed by Task 2.
- Produces: `app.config.INSTRUCTIONS_FILE: Path` — path to
  `<repo_root>/backend/instructions.txt`. Consumed by `app.instructions`.

- [ ] **Step 1: Add the config constant**

In `backend/app/config.py`, after the existing `CONTENT_DIR` line, add:

```python
INSTRUCTIONS_FILE = REPO_ROOT / "backend" / "instructions.txt"
```

- [ ] **Step 2: Write the failing tests**

Create `backend/tests/test_instructions.py`:

```python
# backend/tests/test_instructions.py
from app import instructions


def test_returns_empty_string_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(instructions, "INSTRUCTIONS_FILE", tmp_path / "instructions.txt")

    assert instructions.get_custom_instructions() == ""


def test_returns_stripped_file_contents(tmp_path, monkeypatch):
    path = tmp_path / "instructions.txt"
    path.write_text("  Keep answers short.  \n", encoding="utf-8")
    monkeypatch.setattr(instructions, "INSTRUCTIONS_FILE", path)

    assert instructions.get_custom_instructions() == "Keep answers short."


def test_returns_empty_string_when_file_is_whitespace_only(tmp_path, monkeypatch):
    path = tmp_path / "instructions.txt"
    path.write_text("   \n\n  ", encoding="utf-8")
    monkeypatch.setattr(instructions, "INSTRUCTIONS_FILE", path)

    assert instructions.get_custom_instructions() == ""
```

- [ ] **Step 3: Run tests to verify they fail**

Run (from `backend/`): `.venv\Scripts\python.exe -m pytest tests/test_instructions.py -v`
Expected: FAIL/ERROR — `app.instructions` module does not exist yet.

- [ ] **Step 4: Implement the module**

Create `backend/app/instructions.py`:

```python
# backend/app/instructions.py
from app.config import INSTRUCTIONS_FILE


def get_custom_instructions() -> str:
    try:
        content = INSTRUCTIONS_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    return content.strip()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_instructions.py -v`
Expected: 3 passed.

- [ ] **Step 6: Add the example file and gitignore entry**

Create `backend/instructions.txt.example`:

```
Keep answers to 2-3 sentences.
Only answer questions about Two Owls Tavern; politely decline anything else.
```

In `.gitignore`, add a new line under the existing `.env` line:

```
backend/instructions.txt
```

- [ ] **Step 7: Commit**

```bash
git add backend/app/config.py backend/app/instructions.py backend/instructions.txt.example backend/tests/test_instructions.py .gitignore
git commit -m "Add operator-tunable custom instructions reader"
```

---

### Task 2: Wire custom instructions into the generate node

**Files:**
- Modify: `backend/app/workflow.py` (`_generate_node`, plus the module-level import)
- Test: `backend/tests/test_workflow.py`

**Interfaces:**
- Consumes: `app.instructions.get_custom_instructions() -> str` from Task 1, imported
  into `workflow.py`'s module namespace as `get_custom_instructions` so tests can
  `monkeypatch.setattr(workflow, "get_custom_instructions", ...)`.

- [ ] **Step 1: Extend the test double to capture call kwargs**

In `backend/tests/test_workflow.py`, modify `_ScriptedChatAPI.__init__` and `.create`
to record every call's kwargs (needed to inspect the generate-node system prompt):

```python
class _ScriptedChatAPI:
    def __init__(self, contents: list):
        self._queue = list(contents)
        self.call_count = 0
        self.call_kwargs: list[dict] = []
        # Texts handed to embed_texts, in order -- populated by _patch_common.
        self.embed_inputs: list[list[str]] = []

    def create(self, **kwargs):
        self.call_count += 1
        self.call_kwargs.append(kwargs)
        content = self._queue.pop(0)
        if isinstance(content, Exception):
            raise content
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
```

(Only the two added lines — `self.call_kwargs: list[dict] = []` in `__init__` and
`self.call_kwargs.append(kwargs)` in `create` — are new; everything else in the class
is unchanged from the current file.)

- [ ] **Step 2: Write the failing tests**

Add to `backend/tests/test_workflow.py`:

```python
def test_generate_appends_custom_instructions_when_present(monkeypatch):
    fake_chat_api = _patch_common(monkeypatch, ["general", "first answer", "pass"])
    monkeypatch.setattr(workflow, "get_custom_instructions", lambda: "Keep answers short.")

    workflow.run_chat_workflow(tenant_id=1, question="What do you do?")

    # order: classify, generate, critique -- generate is call index 1
    system_prompt = fake_chat_api.call_kwargs[1]["messages"][0]["content"]
    assert "ADDITIONAL OPERATOR INSTRUCTIONS" in system_prompt
    assert "Keep answers short." in system_prompt


def test_generate_omits_instructions_block_when_none_configured(monkeypatch):
    fake_chat_api = _patch_common(monkeypatch, ["general", "first answer", "pass"])
    monkeypatch.setattr(workflow, "get_custom_instructions", lambda: "")

    workflow.run_chat_workflow(tenant_id=1, question="What do you do?")

    system_prompt = fake_chat_api.call_kwargs[1]["messages"][0]["content"]
    assert "ADDITIONAL OPERATOR INSTRUCTIONS" not in system_prompt
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_workflow.py -v`
Expected: the two new tests FAIL with `AttributeError: <module 'app.workflow'> does
not have the attribute 'get_custom_instructions'` (monkeypatch target doesn't exist
yet). The 6 pre-existing tests still PASS.

- [ ] **Step 4: Implement the wiring**

In `backend/app/workflow.py`, add the import near the top (alongside the existing
`app.*` imports):

```python
from app.instructions import get_custom_instructions
```

Replace `_generate_node` with:

```python
def _generate_node(state: ChatState) -> dict:
    context = "\n\n".join(state["chunks"]) or "(no matching restaurant information found)"
    system_prompt = (
        "You are the AI assistant for a restaurant, answering as the "
        "restaurant itself (first person plural -- 'we' / 'our'). Answer "
        "grounded only in the provided context, and say when something "
        "isn't covered by it. Never assert with certainty that a dish is "
        "safe for a given allergy -- defer to asking staff directly.\n\n"
        f"--- CONTEXT ---\n{context}"
    )
    custom_instructions = get_custom_instructions()
    if custom_instructions:
        system_prompt += (
            f"\n\n--- ADDITIONAL OPERATOR INSTRUCTIONS ---\n{custom_instructions}"
        )
    completion = _client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["question"]},
        ],
    )
    return {"answer": completion.choices[0].message.content or ""}
```

- [ ] **Step 5: Run the full test suite to verify everything passes**

Run (from `backend/`): `.venv\Scripts\python.exe -m pytest -v`
Expected: all tests pass (6 pre-existing `test_workflow.py` tests + 2 new ones + 3 from
`test_instructions.py` = 11 total).

- [ ] **Step 6: Manual smoke test**

With the backend and frontend dev servers running (see `README.md` / CLAUDE.md
commands), create `backend/instructions.txt` with one line, e.g.
`Answer in one sentence only.`, ask a question in the chat widget, confirm the
response is a single sentence. Then empty or delete the file and ask again, confirming
the response reverts to normal length without restarting the backend.

- [ ] **Step 7: Commit**

```bash
git add backend/app/workflow.py backend/tests/test_workflow.py
git commit -m "Append operator custom instructions to the generate-node system prompt"
```
