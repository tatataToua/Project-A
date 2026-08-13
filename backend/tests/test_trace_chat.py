import csv
import json

import pytest

from app import trace_chat

RECORD = {
    "timestamp": "2026-01-01T00:00:00+00:00",
    "question": "when do you open?",
    "answer": "We open at 5pm.",
    "total_latency_s": 3.5,
    "node_latency_s": {"classify": 0.5, "generate": 3.0},
    "tokens": {"prompt": 120, "completion": 30},
    "first_pass": True,
    "retried": False,
    "critique_verdicts": ["pass"],
    "events": [
        {
            "node": "classify",
            "elapsed_s": 0.5,
            "prompt_tokens": 20,
            "completion_tokens": 2,
            "preview": "category=faq",
            "full_text": "category=faq",
        },
        {
            "node": "generate",
            "elapsed_s": 3.0,
            "prompt_tokens": 100,
            "completion_tokens": 28,
            "preview": 'answer="We open at 5pm."',
            "full_text": "We open at 5pm.",
        },
    ],
}


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    path = tmp_path / "logs" / "chat_trace.log"
    path.parent.mkdir()
    monkeypatch.setattr(trace_chat, "LOG_PATH", path)
    return path


def test_format_node_line_layout():
    line = trace_chat._format_node_line("classify", 0.5, 20, 2, "category=faq")
    assert line == "  [classify ]  0.50s |   20 in /   2 out tok | category=faq"


def test_format_summary_includes_answer_tokens_and_verdicts():
    summary = trace_chat._format_summary(RECORD)
    assert "answer: We open at 5pm." in summary
    assert "3.50s | 120 in / 30 out tok" in summary
    assert "first-pass" in summary
    assert "critique: pass" in summary


def test_format_summary_marks_retried_turns():
    retried = {**RECORD, "retried": True, "critique_verdicts": ["fail (retrying)", "pass"]}
    summary = trace_chat._format_summary(retried)
    assert "retried" in summary
    assert "critique: fail (retrying), pass" in summary


def test_format_summary_tolerates_empty_record():
    summary = trace_chat._format_summary({})
    assert "answer: " in summary
    assert "0.00s | 0 in / 0 out tok" in summary


def test_format_logged_turn_replays_every_node():
    text = trace_chat._format_logged_turn(RECORD)
    assert text.startswith("You: when do you open?")
    assert "[classify ]" in text and "[generate ]" in text
    assert "answer: We open at 5pm." in text


def test_run_turn_prints_nodes_then_summary(monkeypatch, capsys):
    def fake_trace_turn(tenant_id, question, on_node=None):
        assert (tenant_id, question) == (3, "hi")
        on_node("classify", 0.5, 20, 2, "category=faq")
        return "We open at 5pm.", RECORD

    monkeypatch.setattr(trace_chat, "trace_turn", fake_trace_turn)

    trace_chat._run_turn(3, "hi")

    out = capsys.readouterr().out
    assert "[classify ]" in out
    assert "answer: We open at 5pm." in out


def test_export_csv_writes_one_row_per_node_event(log_path, tmp_path, capsys):
    log_path.write_text(json.dumps(RECORD) + "\n", encoding="utf-8")
    out_path = tmp_path / "out.csv"

    trace_chat._export_csv(out_path)

    rows = list(csv.DictReader(out_path.open(newline="", encoding="utf-8")))
    assert [r["node"] for r in rows] == ["classify", "generate"]
    assert rows[1]["text"] == "We open at 5pm."  # untruncated full_text, not the preview
    assert rows[0]["total_prompt_tokens"] == "120"
    assert rows[0]["question"] == "when do you open?"
    assert "1 turns (2 node events)" in capsys.readouterr().out


def test_export_csv_defaults_next_to_the_log(log_path, capsys):
    log_path.write_text(json.dumps(RECORD) + "\n", encoding="utf-8")

    trace_chat._export_csv(None)

    default_path = log_path.parent / "chat_trace.csv"
    assert default_path.exists()
    assert str(default_path) in capsys.readouterr().out


def test_export_csv_falls_back_to_preview_when_full_text_missing(log_path, tmp_path):
    record = {**RECORD, "events": [{**RECORD["events"][0]}]}
    del record["events"][0]["full_text"]
    log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    out_path = tmp_path / "out.csv"

    trace_chat._export_csv(out_path)

    [row] = list(csv.DictReader(out_path.open(newline="", encoding="utf-8")))
    assert row["text"] == "category=faq"


def test_export_csv_without_log_file(log_path, capsys):
    trace_chat._export_csv(None)
    assert "No trace log yet" in capsys.readouterr().out


def test_export_csv_with_empty_log(log_path, capsys):
    log_path.write_text("\n", encoding="utf-8")
    trace_chat._export_csv(None)
    assert "Trace log is empty." in capsys.readouterr().out


def test_print_stats_aggregates_turns(log_path, capsys):
    retried = {
        **RECORD,
        "first_pass": False,
        "retried": True,
        "total_latency_s": 6.5,
        "tokens": {"prompt": 200, "completion": 50},
        "node_latency_s": {"classify": 1.5, "generate": 5.0},
    }
    log_path.write_text(
        json.dumps(RECORD) + "\n" + json.dumps(retried) + "\n", encoding="utf-8"
    )

    trace_chat._print_stats()

    out = capsys.readouterr().out
    assert "Turns logged:      2" in out
    assert "First-pass rate:   50%" in out
    assert "Avg total latency: 5.00s" in out
    assert "Avg tokens/turn:   200" in out
    assert "classify   1.00s  (n=2)" in out


def test_print_stats_without_log_file(log_path, capsys):
    trace_chat._print_stats()
    assert "No trace log yet" in capsys.readouterr().out


def test_print_stats_with_empty_log(log_path, capsys):
    log_path.write_text("", encoding="utf-8")
    trace_chat._print_stats()
    assert "Trace log is empty." in capsys.readouterr().out


def test_watch_prints_new_lines_then_stops_on_keyboard_interrupt(log_path, monkeypatch, capsys):
    log_path.write_text(json.dumps({**RECORD, "question": "old turn"}) + "\n", encoding="utf-8")

    def append_then_interrupt(_seconds):
        if not appended:
            appended.append(True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**RECORD, "question": "new turn"}) + "\n")
            return
        raise KeyboardInterrupt

    appended: list[bool] = []
    monkeypatch.setattr(trace_chat.time, "sleep", append_then_interrupt)

    trace_chat._watch()

    out = capsys.readouterr().out
    assert "You: new turn" in out
    assert "old turn" not in out  # only turns that land after the watch starts


def test_watch_creates_a_missing_log_and_skips_blank_lines(log_path, monkeypatch, capsys):
    def append(text):
        with log_path.open("a", encoding="utf-8") as f:
            f.write(text)

    writes = iter([lambda: append("\n"), lambda: append(json.dumps(RECORD) + "\n")])

    def next_write(_seconds):
        try:
            next(writes)()
        except StopIteration:
            raise KeyboardInterrupt

    monkeypatch.setattr(trace_chat.time, "sleep", next_write)

    trace_chat._watch()

    assert log_path.exists()
    assert "You: when do you open?" in capsys.readouterr().out


def test_main_dispatches_to_export_csv(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(trace_chat, "_export_csv", lambda path: calls.append(path))
    monkeypatch.setattr(
        "sys.argv", ["trace_chat", "--export-csv", str(tmp_path / "custom.csv")]
    )

    trace_chat.main()

    assert calls == [tmp_path / "custom.csv"]


def test_main_export_csv_without_path_uses_default(monkeypatch):
    calls = []
    monkeypatch.setattr(trace_chat, "_export_csv", lambda path: calls.append(path))
    monkeypatch.setattr("sys.argv", ["trace_chat", "--export-csv"])

    trace_chat.main()

    assert calls == [None]


def test_main_dispatches_to_stats_and_watch(monkeypatch):
    called = []
    monkeypatch.setattr(trace_chat, "_print_stats", lambda: called.append("stats"))
    monkeypatch.setattr(trace_chat, "_watch", lambda: called.append("watch"))

    monkeypatch.setattr("sys.argv", ["trace_chat", "--stats"])
    trace_chat.main()
    monkeypatch.setattr("sys.argv", ["trace_chat", "--watch"])
    trace_chat.main()

    assert called == ["stats", "watch"]


def _stub_session(monkeypatch, tenant):
    class FakeQuery:
        def filter_by(self, **kwargs):
            return self

        def one_or_none(self):
            return tenant

    class FakeSession:
        def query(self, _model):
            return FakeQuery()

        def close(self):
            pass

    monkeypatch.setattr(trace_chat, "SessionLocal", lambda: FakeSession())


def test_main_repl_reports_unknown_tenant(monkeypatch, capsys):
    _stub_session(monkeypatch, None)
    monkeypatch.setattr("sys.argv", ["trace_chat", "nope"])

    trace_chat.main()

    assert "No tenant 'nope' found" in capsys.readouterr().out


def test_main_repl_runs_questions_until_exit(monkeypatch, capsys):
    class FakeTenant:
        id = 42

    _stub_session(monkeypatch, FakeTenant())
    monkeypatch.setattr(trace_chat, "instrument_client", lambda: None)
    turns = []
    monkeypatch.setattr(trace_chat, "_run_turn", lambda tid, q: turns.append((tid, q)))
    answers = iter(["  ", "when do you open?", "EXIT", "never asked"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr("sys.argv", ["trace_chat"])

    trace_chat.main()

    assert turns == [(42, "when do you open?")]
    assert "Tracing chat workflow for tenant 'two-owls-tavern'" in capsys.readouterr().out


def test_main_repl_exits_on_eof(monkeypatch):
    class FakeTenant:
        id = 1

    _stub_session(monkeypatch, FakeTenant())
    monkeypatch.setattr(trace_chat, "instrument_client", lambda: None)
    monkeypatch.setattr(trace_chat, "_run_turn", lambda tid, q: pytest.fail("no turn expected"))

    def raise_eof(_prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)
    monkeypatch.setattr("sys.argv", ["trace_chat"])

    trace_chat.main()
