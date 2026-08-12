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
    monkeypatch.setattr(workflow, "retrieve_chunks", lambda tenant_id, vec, **kw: ["some background chunk"])
    monkeypatch.setattr(workflow, "get_custom_instructions", lambda: "")
    return fake_chat_api


def test_answers_directly_when_critique_passes(monkeypatch):
    # order: classify, generate, critique
    fake_chat_api = _patch_common(monkeypatch, ["general", "first answer", "pass"])

    answer = workflow.run_chat_workflow(tenant_id=1, question="What do you do?")

    assert answer == "first answer"
    assert fake_chat_api.call_count == 3


def test_retries_exactly_once_when_critique_fails(monkeypatch):
    # order: classify, generate, critique(fail), generate(retry), (critique skipped on retry)
    fake_chat_api = _patch_common(
        monkeypatch, ["hours_location", "first answer", "fail", "second answer"]
    )

    answer = workflow.run_chat_workflow(tenant_id=1, question="Where did you go to school?")

    assert answer == "second answer"
    assert fake_chat_api.call_count == 4


def test_retrieval_search_text_is_biased_by_classified_category(monkeypatch):
    fake_chat_api = _patch_common(monkeypatch, ["hours_location", "first answer", "pass"])

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


def test_critique_failure_keeps_the_existing_answer(monkeypatch):
    # classify, generate, critique raises -> no retry, answer stands
    fake_chat_api = _patch_common(
        monkeypatch, ["general", "first answer", OpenAIError("critique down")]
    )

    answer = workflow.run_chat_workflow(tenant_id=1, question="What do you do?")

    assert answer == "first answer"
    assert fake_chat_api.call_count == 3


def test_generate_appends_custom_instructions_when_present(monkeypatch):
    fake_chat_api = _patch_common(monkeypatch, ["general", "first answer", "pass"])
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
    fake_chat_api = _patch_common(monkeypatch, ["general", "first answer", "pass"])
    monkeypatch.setattr(workflow, "get_custom_instructions", lambda: "")

    workflow.run_chat_workflow(tenant_id=1, question="What do you do?")

    system_prompt = fake_chat_api.call_kwargs[1]["messages"][0]["content"]
    assert "ADDITIONAL OPERATOR INSTRUCTIONS" not in system_prompt
