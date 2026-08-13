"""Covers the golden-set eval gate's own logic -- dataset construction and the
pass/fail thresholds -- with retrieval, the workflow, and the Ragas judge all
stubbed out, so no LLM, Postgres, or judge model is involved."""
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from app import eval_rag

GOLDEN = [
    {"question": "When do you open?", "expected_answer": "5pm"},
    {"question": "What year did you open?", "expected_answer": 2015.0},
]


def write_golden_set(tmp_path, monkeypatch, items, tenant="two-owls-tavern"):
    path = tmp_path / "tenants" / tenant / "eval"
    path.mkdir(parents=True)
    (path / "golden_qa.yaml").write_text(yaml.safe_dump(items), encoding="utf-8")
    monkeypatch.setattr(eval_rag, "CONTENT_DIR", tmp_path)
    return path / "golden_qa.yaml"


def stub_pipeline(monkeypatch, *, answers=None):
    """Stub out the tenant lookup and the retrieval/generation pipeline."""
    class FakeSession:
        def query(self, _model):
            return self

        def filter_by(self, **kwargs):
            return self

        def one(self):
            return SimpleNamespace(id=9)

        def close(self):
            pass

    monkeypatch.setattr(eval_rag, "SessionLocal", lambda: FakeSession())
    monkeypatch.setattr(eval_rag, "embed_texts", lambda texts: [[0.1] * 3 for _ in texts])
    monkeypatch.setattr(
        eval_rag,
        "retrieve_chunks",
        lambda tenant_id, question, vector: [("about.md", "we open at 5pm")],
    )
    replies = iter(answers or ["We open at 5pm.", 2015.0])
    monkeypatch.setattr(eval_rag, "run_chat_workflow", lambda tenant_id, question: next(replies))


def test_load_golden_set_reads_yaml(tmp_path, monkeypatch):
    write_golden_set(tmp_path, monkeypatch, GOLDEN)
    assert eval_rag.load_golden_set("two-owls-tavern") == GOLDEN


def test_load_golden_set_missing_file_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_rag, "CONTENT_DIR", tmp_path)
    with pytest.raises(SystemExit, match="No golden eval set"):
        eval_rag.load_golden_set("two-owls-tavern")


def test_build_dataset_coerces_non_string_yaml_scalars(monkeypatch, capsys):
    stub_pipeline(monkeypatch)

    dataset = eval_rag.build_dataset("two-owls-tavern", GOLDEN)

    assert dataset["question"] == [item["question"] for item in GOLDEN]
    assert dataset["answer"] == ["We open at 5pm.", "2015.0"]
    assert dataset["reference"] == ["5pm", "2015.0"]
    assert dataset["contexts"] == [["we open at 5pm"], ["we open at 5pm"]]
    assert "[1/2] When do you open?" in capsys.readouterr().out


def stub_judge(monkeypatch, scores):
    monkeypatch.setattr(eval_rag, "ChatOpenAI", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(
        eval_rag,
        "evaluate",
        lambda dataset, metrics, llm, run_config: SimpleNamespace(
            to_pandas=lambda: pd.DataFrame({"faithfulness": scores})
        ),
    )


def run_main(monkeypatch, tmp_path, scores, items=None):
    items = items or GOLDEN
    write_golden_set(tmp_path, monkeypatch, items)
    stub_pipeline(monkeypatch, answers=["answer"] * len(items))
    stub_judge(monkeypatch, scores)
    monkeypatch.setattr("sys.argv", ["eval_rag", "--tenant", "two-owls-tavern"])
    eval_rag.main()


def test_main_passes_when_faithfulness_above_threshold(monkeypatch, tmp_path, capsys):
    run_main(monkeypatch, tmp_path, [0.9, 1.0])
    assert "PASSED: faithfulness above threshold." in capsys.readouterr().out


def test_main_fails_when_faithfulness_below_threshold(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, tmp_path, [0.1, 0.2])
    assert exc.value.code == 1
    assert "below threshold" in capsys.readouterr().out


def test_main_fails_when_every_item_failed_to_score(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, tmp_path, [float("nan"), float("nan")])
    assert exc.value.code == 1
    assert "treated as an infrastructure failure" in capsys.readouterr().out


def test_main_fails_when_too_many_items_failed_to_score(monkeypatch, tmp_path, capsys):
    # 1 of 5 unscored = 20%, over the 10% isolated-flake tolerance.
    items = [{"question": f"q{i}", "expected_answer": "a"} for i in range(5)]
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, tmp_path, [1.0, 1.0, 1.0, 1.0, float("nan")], items=items)
    assert exc.value.code == 1
    assert "exceeds the" in capsys.readouterr().out


def test_main_tolerates_isolated_judge_flake(monkeypatch, tmp_path, capsys):
    # 1 of 10 unscored = exactly the 10% tolerance: mean over the other 9 still gates.
    items = [{"question": f"q{i}", "expected_answer": "a"} for i in range(10)]
    run_main(monkeypatch, tmp_path, [1.0] * 9 + [float("nan")], items=items)

    out = capsys.readouterr().out
    assert "Scored: 9/10" in out
    assert "1/10 questions failed to score" in out
    assert "PASSED" in out
