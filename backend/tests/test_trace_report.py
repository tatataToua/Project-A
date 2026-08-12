import pytest

from app import trace_report


def test_percentile_empty_returns_zero():
    assert trace_report._percentile([], 50) == 0.0


def test_percentile_p50_and_p95():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert trace_report._percentile(values, 50) == 3.0
    assert trace_report._percentile(values, 95) == 5.0


def test_estimate_cost_known_model():
    record = {"tokens": {"prompt": 1000, "completion": 1000}}
    cost = trace_report.estimate_cost(record, "gemini-flash-latest")
    pricing = trace_report.PRICING_PER_1K_TOKENS["gemini-flash-latest"]
    assert cost == pytest.approx(pricing["prompt"] + pricing["completion"])


def test_estimate_cost_unknown_model_is_zero():
    record = {"tokens": {"prompt": 1000, "completion": 1000}}
    assert trace_report.estimate_cost(record, "qwen2.5:7b-instruct") == 0.0


def test_is_declined_true_when_last_verdict_is_declined():
    record = {"critique_verdicts": ["fail (retrying)", "fail (declined)"]}
    assert trace_report.is_declined(record) is True


def test_is_declined_false_when_passed():
    record = {"critique_verdicts": ["pass"]}
    assert trace_report.is_declined(record) is False


def test_is_declined_false_when_no_verdicts():
    assert trace_report.is_declined({"critique_verdicts": []}) is False


def test_build_report_aggregates_correctly():
    records = [
        {
            "total_latency_s": 1.0,
            "tokens": {"prompt": 100, "completion": 100},
            "critique_verdicts": ["pass"],
            "retried": False,
        },
        {
            "total_latency_s": 2.0,
            "tokens": {"prompt": 100, "completion": 100},
            "critique_verdicts": ["fail (retrying)", "pass"],
            "retried": True,
        },
        {
            "total_latency_s": 3.0,
            "tokens": {"prompt": 100, "completion": 100},
            "critique_verdicts": ["fail (retrying)", "fail (declined)"],
            "retried": True,
        },
    ]
    report = trace_report.build_report(records, "unknown-model")

    assert report["n"] == 3
    assert report["latency_p50_s"] == 2.0
    assert report["citation_coverage_pct"] == pytest.approx(200 / 3)
    assert report["failure_rate_pct"] == pytest.approx(100 / 3)
    assert report["retry_rate_pct"] == pytest.approx(200 / 3)


def test_load_records_reads_jsonl(tmp_path):
    log_path = tmp_path / "chat_trace.log"
    log_path.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    records = trace_report.load_records(log_path)
    assert records == [{"a": 1}, {"a": 2}]


def test_load_records_missing_file_returns_empty(tmp_path):
    assert trace_report.load_records(tmp_path / "does-not-exist.log") == []
