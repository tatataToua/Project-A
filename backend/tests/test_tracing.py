import json
from types import SimpleNamespace

import pytest

from app import tracing


class FakeGraph:
    """Stands in for the LangGraph workflow graph: yields one `{node: delta}`
    update per node, the same shape `.stream(stream_mode="updates")` produces."""

    def __init__(self, updates):
        self._updates = updates
        self.streamed_state = None

    def stream(self, state, stream_mode=None):
        self.streamed_state = state
        assert stream_mode == "updates"
        for update in self._updates:
            yield update


@pytest.fixture
def isolated_log(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "chat_trace.log"
    monkeypatch.setattr(tracing, "LOG_PATH", log_path)
    return log_path


@pytest.fixture(autouse=True)
def clean_usage_buffer():
    tracing._usage_buffer.clear()
    yield
    tracing._usage_buffer.clear()


def test_preview_for_node_per_node_shapes():
    assert tracing._preview_for_node("classify", {"category": "menu"}) == "category=menu"
    assert tracing._preview_for_node("retrieve", {"chunks": ["a", "b"]}) == "2 chunks retrieved"
    assert tracing._preview_for_node("generate", {"answer": "hello"}) == 'answer="hello"'
    assert tracing._preview_for_node("critique", {"needs_retry": True}) == "fail (retrying)"
    assert tracing._preview_for_node("critique", {"answer": "declined"}) == "fail (declined)"
    assert tracing._preview_for_node("critique", {}) == "pass"
    assert tracing._preview_for_node("unknown", {"x": 1}) == "{'x': 1}"


def test_preview_for_generate_truncates_long_answers():
    preview = tracing._preview_for_node("generate", {"answer": "x" * 100})
    assert preview == f'answer="{"x" * 70}..."'


def test_preview_for_generate_handles_missing_answer():
    assert tracing._preview_for_node("generate", {"answer": None}) == 'answer=""'


def test_full_text_for_node_expands_generate_and_retrieve():
    assert tracing._full_text_for_node("generate", {"answer": "x" * 100}, "truncated") == "x" * 100
    assert (
        tracing._full_text_for_node("retrieve", {"search_text": "hours"}, "2 chunks retrieved")
        == '2 chunks retrieved (embedded query: "hours")'
    )
    assert tracing._full_text_for_node("classify", {}, "category=menu") == "category=menu"


def test_instrument_client_records_usage_and_is_idempotent(monkeypatch):
    chat_calls = []
    embed_calls = []

    def chat_create(**kwargs):
        chat_calls.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7))

    def embed_create(**kwargs):
        embed_calls.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=5))

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=chat_create)),
        embeddings=SimpleNamespace(create=embed_create),
    )
    monkeypatch.setattr("app.llm.client", fake_client)
    monkeypatch.setattr(tracing, "_instrumented", False)

    tracing.instrument_client()
    wrapped_chat = fake_client.chat.completions.create
    tracing.instrument_client()
    assert fake_client.chat.completions.create is wrapped_chat  # not double-wrapped

    fake_client.chat.completions.create(model="m")
    fake_client.embeddings.create(model="e")

    assert tracing._usage_buffer == [
        {"prompt_tokens": 11, "completion_tokens": 7},
        {"prompt_tokens": 5, "completion_tokens": 0},
    ]
    assert chat_calls == [{"model": "m"}] and embed_calls == [{"model": "e"}]


def test_instrument_client_tolerates_missing_usage(monkeypatch):
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace())),
        embeddings=SimpleNamespace(create=lambda **kw: SimpleNamespace(usage=None)),
    )
    monkeypatch.setattr("app.llm.client", fake_client)
    monkeypatch.setattr(tracing, "_instrumented", False)
    tracing.instrument_client()

    fake_client.chat.completions.create()
    fake_client.embeddings.create()

    assert tracing._usage_buffer == [
        {"prompt_tokens": 0, "completion_tokens": 0},
        {"prompt_tokens": 0, "completion_tokens": 0},
    ]


def test_trace_turn_records_nodes_tokens_and_log_line(isolated_log, monkeypatch):
    graph = FakeGraph(
        [
            {"classify": {"category": "menu", "search_text": "hours"}},
            {"retrieve": {"chunks": ["c1", "c2"], "search_text": "hours"}},
            {"generate": {"answer": "We open at 5pm."}},
            {"critique": {}},
        ]
    )
    monkeypatch.setattr(tracing, "_graph", graph)

    answer, record = tracing.trace_turn(7, "when do you open?", on_node=None)

    assert answer == "We open at 5pm."
    assert graph.streamed_state["tenant_id"] == 7
    assert graph.streamed_state["question"] == "when do you open?"
    assert [e["node"] for e in record["events"]] == ["classify", "retrieve", "generate", "critique"]
    assert record["critique_verdicts"] == ["pass"]
    assert record["first_pass"] is True
    assert record["retried"] is False
    assert record["answer"] == "We open at 5pm."
    assert set(record["node_latency_s"]) == {"classify", "retrieve", "generate", "critique"}

    logged = [json.loads(line) for line in isolated_log.read_text(encoding="utf-8").splitlines()]
    assert len(logged) == 1
    assert logged[0]["question"] == "when do you open?"


def test_trace_turn_attributes_token_usage_per_node(isolated_log, monkeypatch):
    class TokenSpendingGraph:
        def stream(self, state, stream_mode=None):
            tracing._usage_buffer.append({"prompt_tokens": 10, "completion_tokens": 2})
            yield {"classify": {"category": "menu"}}
            tracing._usage_buffer.append({"prompt_tokens": 100, "completion_tokens": 40})
            yield {"generate": {"answer": "answer"}}
            yield {"critique": {"needs_retry": True}}

    monkeypatch.setattr(tracing, "_graph", TokenSpendingGraph())

    _answer, record = tracing.trace_turn(1, "q")

    assert record["node_tokens"]["classify"] == {"prompt": 10, "completion": 2}
    assert record["node_tokens"]["generate"] == {"prompt": 100, "completion": 40}
    assert record["node_tokens"]["critique"] == {"prompt": 0, "completion": 0}
    assert record["tokens"] == {"prompt": 110, "completion": 42}
    assert record["critique_verdicts"] == ["fail (retrying)"]
    assert record["first_pass"] is False


def test_trace_turn_sums_repeated_node_visits_on_retry(isolated_log, monkeypatch):
    graph = FakeGraph(
        [
            {"generate": {"answer": "first"}},
            {"critique": {"needs_retry": True}},
            {"generate": {"answer": "second"}},
            {"critique": {}},
        ]
    )
    monkeypatch.setattr(tracing, "_graph", graph)

    answer, record = tracing.trace_turn(1, "q")

    assert answer == "second"
    assert record["critique_verdicts"] == ["fail (retrying)", "pass"]
    assert record["first_pass"] is False
    assert len(record["events"]) == 4
    assert set(record["node_latency_s"]) == {"generate", "critique"}


def test_trace_turn_invokes_on_node_callback(isolated_log, monkeypatch):
    monkeypatch.setattr(
        tracing, "_graph", FakeGraph([{"classify": {"category": "faq"}}, {"critique": {}}])
    )
    seen = []

    tracing.trace_turn(1, "q", on_node=lambda *args: seen.append(args))

    assert [(name, preview) for name, _elapsed, _p, _c, preview in seen] == [
        ("classify", "category=faq"),
        ("critique", "pass"),
    ]


def test_trace_turn_appends_to_existing_log(isolated_log, monkeypatch):
    monkeypatch.setattr(tracing, "_graph", FakeGraph([{"generate": {"answer": "a"}}]))

    tracing.trace_turn(1, "first")
    tracing.trace_turn(1, "second")

    lines = isolated_log.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["question"] for line in lines] == ["first", "second"]
