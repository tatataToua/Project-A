# Classify-Node Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `_classify_node` from a freeform string response (validated by checking
membership in a hardcoded tuple) to a Pydantic-validated JSON schema response, with one
retry on invalid output before falling back to `"general"`. This is the second half of
sub-project #6 of the Phase 3.5 design spec — "constrained generation + validation +
retry," applied to code that already exists rather than a standalone demo.

**Architecture:** `_classify_node` asks the LLM for `{"category": "<...>"}` instead of a
bare word, parses the response with `json.loads` + a new `_ClassifyResult` Pydantic
model (`Literal["menu", "hours_location", "policies", "general"]`), and on invalid
output (malformed JSON, or JSON that doesn't match the schema) re-prompts once with a
corrective message before falling back to `"general"` — the same fallback already used
for an API failure. `pydantic` is already a dependency (via `fastapi`, already imported
directly in `main.py`) — no new dependency needed.

**Depends on:** the citation-enforcement plan must be merged first — this task modifies
the same functions in `workflow.py`/`test_workflow.py` that plan already changed
(`_classify_node`, and every test's scripted `chat_contents` list, since the classify
response shape is changing from a bare string to JSON).

## Global Constraints

- No new dependencies — `pydantic` is already available.
- Windows dev environment: run backend commands via `C:\pyvenvs\p35backend\Scripts\python.exe`.
- The retry-on-invalid-output pattern is exactly one retry (2 total attempts), matching
  this codebase's existing "one retry max" philosophy (the self-critique retry loop is
  also exactly one retry) — do not add configurable retry counts or backoff.

---

## Task 1: Pydantic-validated classify node

**Files:**
- Modify: `backend/app/workflow.py`
- Modify: `backend/tests/test_workflow.py`

**Interfaces:**
- No change to `_classify_node`'s external contract — still returns `{"category": str}`
  where the string is one of `menu`/`hours_location`/`policies`/`general`. The change is
  entirely in how that value gets produced and validated.

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `backend/tests/test_workflow.py`:

```python
# backend/tests/test_workflow.py
from types import SimpleNamespace

from openai import OpenAIError

from app import workflow


class _ScriptedChatAPI:
    """Returns queued responses in order; each call consumes the next one.

    A queued entry that is an exception instance is raised instead of returned,
    which is how the LLM-failure fallbacks are exercised.
    """

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


def _patch_common(monkeypatch, chat_contents: list):
    fake_chat_api = _ScriptedChatAPI(chat_contents)
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=fake_chat_api))

    def fake_embed_texts(texts):
        fake_chat_api.embed_inputs.append(list(texts))
        return [[0.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(workflow, "_client", fake_client)
    monkeypatch.setattr(workflow, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(workflow, "retrieve_chunks", lambda tenant_id, query_text, vec, **kw: [("bio.md", "some background chunk")])
    monkeypatch.setattr(workflow, "get_custom_instructions", lambda: "")
    return fake_chat_api


def test_answers_directly_when_critique_passes(monkeypatch):
    # order: classify, generate, critique
    fake_chat_api = _patch_common(monkeypatch, ['{"category": "general"}', "first answer", "pass"])

    answer = workflow.run_chat_workflow(tenant_id=1, question="What do you do?")

    assert answer == "first answer"
    assert fake_chat_api.call_count == 3


def test_retries_exactly_once_when_critique_fails(monkeypatch):
    # order: classify, generate, critique(fail), generate(retry), critique(pass)
    fake_chat_api = _patch_common(
        monkeypatch, ['{"category": "hours_location"}', "first answer", "fail", "second answer", "pass"]
    )

    answer = workflow.run_chat_workflow(tenant_id=1, question="Where did you go to school?")

    assert answer == "second answer"
    assert fake_chat_api.call_count == 5


def test_declines_when_retried_answer_still_fails_critique(monkeypatch):
    # order: classify, generate, critique(fail), generate(retry), critique(fail again)
    fake_chat_api = _patch_common(
        monkeypatch, ['{"category": "general"}', "first answer", "fail", "second answer", "fail"]
    )

    answer = workflow.run_chat_workflow(tenant_id=1, question="What do you do?")

    assert answer == workflow.REFUSAL_MESSAGE
    assert fake_chat_api.call_count == 5


def test_generate_context_includes_source_attribution(monkeypatch):
    fake_chat_api = _patch_common(monkeypatch, ['{"category": "general"}', "first answer", "pass"])
    monkeypatch.setattr(
        workflow, "retrieve_chunks",
        lambda tenant_id, query_text, vec, **kw: [("menu.md", "Margherita pizza — $17")],
    )

    workflow.run_chat_workflow(tenant_id=1, question="What pizzas do you have?")

    system_prompt = fake_chat_api.call_kwargs[1]["messages"][0]["content"]
    assert "[Source: menu.md]" in system_prompt
    assert "Margherita pizza — $17" in system_prompt


def test_retrieval_search_text_is_biased_by_classified_category(monkeypatch):
    fake_chat_api = _patch_common(monkeypatch, ['{"category": "hours_location"}', "first answer", "pass"])

    workflow.run_chat_workflow(tenant_id=1, question="Where did you go to school?")

    [search_text] = fake_chat_api.embed_inputs[0]
    assert search_text == "[hours_location] Where did you go to school?"


def test_classify_failure_degrades_to_general_and_still_answers(monkeypatch):
    # classify raises, then: generate, critique
    fake_chat_api = _patch_common(
        monkeypatch, [OpenAIError("classify down"), "first answer", "pass"]
    )

    answer = workflow.run_chat_workflow(tenant_id=1, question="What do you do?")

    assert answer == "first answer"
    assert fake_chat_api.call_count == 3
    [search_text] = fake_chat_api.embed_inputs[0]
    assert search_text == "[general] What do you do?"


def test_classify_node_returns_general_when_llm_fails(monkeypatch):
    _patch_common(monkeypatch, [OpenAIError("classify down")])

    assert workflow._classify_node({"question": "anything"}) == {"category": "general"}


def test_classify_retries_once_on_invalid_json_then_succeeds(monkeypatch):
    fake_chat_api = _patch_common(
        monkeypatch, ["not valid json at all", '{"category": "menu"}', "first answer", "pass"]
    )

    answer = workflow.run_chat_workflow(tenant_id=1, question="What's on the menu?")

    assert answer == "first answer"
    assert fake_chat_api.call_count == 4
    [search_text] = fake_chat_api.embed_inputs[0]
    assert search_text == "[menu] What's on the menu?"


def test_classify_falls_back_to_general_after_two_invalid_attempts(monkeypatch):
    fake_chat_api = _patch_common(
        monkeypatch, ["not valid json", "still not valid json", "first answer", "pass"]
    )

    answer = workflow.run_chat_workflow(tenant_id=1, question="What do you do?")

    assert answer == "first answer"
    assert fake_chat_api.call_count == 4
    [search_text] = fake_chat_api.embed_inputs[0]
    assert search_text == "[general] What do you do?"


def test_parse_classify_response_accepts_valid_category():
    assert workflow._parse_classify_response('{"category": "policies"}') == "policies"


def test_parse_classify_response_rejects_malformed_json():
    assert workflow._parse_classify_response("not json") is None


def test_parse_classify_response_rejects_unknown_category():
    assert workflow._parse_classify_response('{"category": "not_a_real_category"}') is None


def test_parse_classify_response_rejects_missing_field():
    assert workflow._parse_classify_response('{"not_category": "menu"}') is None


def test_parse_classify_response_rejects_none():
    assert workflow._parse_classify_response(None) is None


def test_critique_failure_keeps_the_existing_answer(monkeypatch):
    # classify, generate, critique raises -> no retry, answer stands
    fake_chat_api = _patch_common(
        monkeypatch, ['{"category": "general"}', "first answer", OpenAIError("critique down")]
    )

    answer = workflow.run_chat_workflow(tenant_id=1, question="What do you do?")

    assert answer == "first answer"
    assert fake_chat_api.call_count == 3


def test_generate_appends_custom_instructions_when_present(monkeypatch):
    fake_chat_api = _patch_common(monkeypatch, ['{"category": "general"}', "first answer", "pass"])
    monkeypatch.setattr(workflow, "get_custom_instructions", lambda: "Keep answers short.")

    workflow.run_chat_workflow(tenant_id=1, question="What do you do?")

    # order: classify, generate, critique -- generate is call index 1
    system_prompt = fake_chat_api.call_kwargs[1]["messages"][0]["content"]
    assert "ADDITIONAL OPERATOR INSTRUCTIONS" in system_prompt
    assert "Keep answers short." in system_prompt
    assert system_prompt.index("--- CONTEXT ---") < system_prompt.index(
        "--- ADDITIONAL OPERATOR INSTRUCTIONS ---"
    )


def test_generate_omits_instructions_block_when_none_configured(monkeypatch):
    fake_chat_api = _patch_common(monkeypatch, ['{"category": "general"}', "first answer", "pass"])
    monkeypatch.setattr(workflow, "get_custom_instructions", lambda: "")

    workflow.run_chat_workflow(tenant_id=1, question="What do you do?")

    system_prompt = fake_chat_api.call_kwargs[1]["messages"][0]["content"]
    assert "ADDITIONAL OPERATOR INSTRUCTIONS" not in system_prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\pyvenvs\p35backend\Scripts\python.exe -m pytest tests/test_workflow.py -v` (from `backend/`)
Expected: FAIL — most tests fail because `_classify_node` still returns the raw string
content as the category (e.g. `'{"category": "general"}'` itself becomes the "category",
which isn't in the old hardcoded tuple, so it falls back to `"general"` — this breaks
category-dependent assertions like the `[hours_location]` search-text checks). The new
`_parse_classify_response`/retry tests fail with `AttributeError`/`ModuleNotFoundError`
since that function doesn't exist yet.

- [ ] **Step 3: Implement `app/workflow.py`**

Change the imports at the top of the file from:
```python
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAIError

from app.config import GEMINI_MODEL
from app.embeddings import embed_texts
from app.instructions import get_custom_instructions
from app.llm import client as _client
from app.retrieval import retrieve_chunks

REFUSAL_MESSAGE = "We don't have that information on hand — please ask a staff member directly."
```
to:
```python
import json
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAIError
from pydantic import BaseModel, ValidationError

from app.config import GEMINI_MODEL
from app.embeddings import embed_texts
from app.instructions import get_custom_instructions
from app.llm import client as _client
from app.retrieval import retrieve_chunks

REFUSAL_MESSAGE = "We don't have that information on hand — please ask a staff member directly."

_CLASSIFY_SYSTEM_PROMPT = (
    "Classify the user's question into exactly one category:\n"
    "'menu' - dishes, prices, ingredients, allergens, drinks.\n"
    "'hours_location' - hours, address, parking, reservations.\n"
    "'policies' - dress code, gratuity/split-check, dietary "
    "accommodation, pets, private events, gift cards, holiday closures.\n"
    "'general' - anything else.\n"
    'Respond with ONLY a JSON object of the form {"category": "<one of the above>"} '
    "-- no other text."
)


class _ClassifyResult(BaseModel):
    category: Literal["menu", "hours_location", "policies", "general"]
```

Replace `_classify_node` in full (it currently spans from `def _classify_node` down to
just before `def _retrieve_node`):

```python
def _parse_classify_response(content: str | None) -> str | None:
    """Returns the validated category, or None if content isn't valid JSON
    matching the schema."""
    if not content:
        return None
    try:
        data = json.loads(content)
        result = _ClassifyResult.model_validate(data)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None
    return result.category


def _classify_node(state: ChatState) -> dict:
    messages = [
        {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
        {"role": "user", "content": state["question"]},
    ]
    for _attempt in range(2):
        try:
            completion = _client.chat.completions.create(model=GEMINI_MODEL, messages=messages)
        except OpenAIError:
            # Classification only biases the retrieval query -- nothing depends on it,
            # so degrade to the same "general" fallback used for an unusable answer.
            return {"category": "general"}
        content = completion.choices[0].message.content
        category = _parse_classify_response(content)
        if category is not None:
            return {"category": category}
        # Invalid output -- retry once with a corrective nudge before falling back.
        messages = messages + [
            {"role": "assistant", "content": content or ""},
            {
                "role": "user",
                "content": (
                    "That wasn't valid JSON matching the schema. Respond with ONLY "
                    'the JSON object, e.g. {"category": "menu"}.'
                ),
            },
        ]
    return {"category": "general"}
```

Everything else in `workflow.py` (`_retrieve_node` onward) stays unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\pyvenvs\p35backend\Scripts\python.exe -m pytest tests/test_workflow.py -v` (from `backend/`)
Expected: all tests PASS, including the 7 new/changed ones.

- [ ] **Step 5: Run the full backend test suite**

Run: `C:\pyvenvs\p35backend\Scripts\python.exe -m pytest -q` (from `backend/`)
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/workflow.py backend/tests/test_workflow.py
git commit -m "Harden classify node: Pydantic-validated JSON output with one retry on invalid response"
```
