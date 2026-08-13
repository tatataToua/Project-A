from types import SimpleNamespace

from app import tracing


def _fake_graph(answer: str = "an answer"):
    def stream(state, stream_mode="updates"):
        yield {"generate": {"answer": answer}}
        yield {"critique": {"needs_retry": False}}

    return SimpleNamespace(stream=stream)


def test_trace_turn_appends_a_record(tmp_path, monkeypatch):
    monkeypatch.setattr(tracing, "_graph", _fake_graph())
    monkeypatch.setattr(tracing, "LOG_PATH", tmp_path / "logs" / "chat_trace.log")

    answer, record = tracing.trace_turn(1, "hi")

    assert answer == "an answer"
    assert record["question"] == "hi"
    assert len(tracing.read_records(tracing.LOG_PATH)) == 1


def test_trace_turn_still_returns_answer_when_log_write_fails(tmp_path, monkeypatch, caplog):
    # A file where the log directory should be makes mkdir/open raise OSError.
    blocking_file = tmp_path / "logs"
    blocking_file.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(tracing, "_graph", _fake_graph())
    monkeypatch.setattr(tracing, "LOG_PATH", blocking_file / "chat_trace.log")

    with caplog.at_level("WARNING", logger="askme"):
        answer, _record = tracing.trace_turn(1, "hi")

    assert answer == "an answer"
    assert "Could not append trace record" in caplog.text


def test_read_records_skips_malformed_lines(tmp_path, caplog):
    log_path = tmp_path / "chat_trace.log"
    log_path.write_text('{"question": "a"}\n{"question": "b"\n{"question": "c"}\n', encoding="utf-8")

    with caplog.at_level("WARNING", logger="askme"):
        records = tracing.read_records(log_path)

    assert [r["question"] for r in records] == ["a", "c"]
    assert "unparseable trace record" in caplog.text


def test_read_records_missing_file_returns_empty(tmp_path):
    assert tracing.read_records(tmp_path / "nope.log") == []
