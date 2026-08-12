# Observability Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Aggregate the existing per-node JSONL tracer (`tracing.py` → `backend/logs/chat_trace.log`) into the metrics that matter for a production RAG system: P50/P95 latency, cost per request, citation coverage, and failure rate. This is sub-project #5 of the Phase 3.5 design spec.

**Architecture:** A new, testable module `app/trace_report.py` (not `scripts/trace_report.py` as the umbrella design spec sketched — this codebase's actual convention, per `app/ingest.py`, is that logic worth unit-testing lives in `app/` with a `python -m app.<name>` CLI entry point, while `scripts/` is reserved for harnesses that aren't unit-tested, like `bench_chat.py`) reads the JSONL log, computes the four metrics, and prints a summary formatted for pasting into `METRICS.md` — the same pattern `bench_chat.py` already established. No new external service.

**Tech Stack:** Standard library only (`json`, `statistics`-adjacent manual percentile calc since `statistics.quantiles` is awkward with small/uneven sample sizes).

**Depends on:** the citation-enforcement plan's critique-verdict format — a declined turn's `critique_verdicts` list ends with the exact string `"fail (declined)"`. If that plan hasn't merged yet, this one can still be implemented and unit-tested (all tests use synthetic record dicts, no real trace log needed) but should not be treated as integration-verified against a real declined turn until both are merged together.

## Global Constraints

- No new dependencies.
- Windows dev environment: run backend commands via `C:\pyvenvs\p35backend\Scripts\python.exe -m pytest` (see the citation-enforcement plan for why this worktree's venv lives outside the repo).
- This module must not require Postgres or a live LLM to test — it's pure aggregation over already-written JSONL, so all tests use synthetic in-memory/tmp-file data.

---

## Task 1: `app/trace_report.py` — aggregation module + CLI

**Files:**
- Create: `backend/app/trace_report.py`
- Test: `backend/tests/test_trace_report.py`

**Interfaces:**
- Produces: `load_records(log_path: Path) -> list[dict]`, `estimate_cost(record: dict, model: str) -> float`, `is_declined(record: dict) -> bool`, `build_report(records: list[dict], model: str) -> dict`, `PRICING_PER_1K_TOKENS: dict` — all standalone, independently testable functions. `main()` is the CLI entry point, not unit-tested directly (it's a thin argument-parsing + print wrapper around the functions above, matching `ingest.py`'s pattern).

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_trace_report.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\pyvenvs\p35backend\Scripts\python.exe -m pytest tests/test_trace_report.py -v` (from `backend/`)
Expected: FAIL with `ModuleNotFoundError: No module named 'app.trace_report'`.

- [ ] **Step 3: Implement `app/trace_report.py`**

```python
"""Aggregates chat_trace.log into P50/P95 latency, cost/request, citation
coverage, and failure rate -- the observability layer for Phase 3.5.
Run on demand; paste the printed summary into METRICS.md.

Usage (from `backend/`):
    .venv\\Scripts\\python.exe -m app.trace_report [--log PATH] [--model NAME]
"""
import argparse
import json
import statistics
from pathlib import Path

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "chat_trace.log"

# Approximate list prices, USD per 1K tokens, as of this writing. A model not
# listed here (e.g. any local Ollama model) is treated as $0/token -- update
# this table if a provider's pricing changes or a new model gets used.
PRICING_PER_1K_TOKENS = {
    "gemini-flash-latest": {"prompt": 0.000075, "completion": 0.0003},
    "gemini-3.1-flash-lite": {"prompt": 0.00005, "completion": 0.0002},
}


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, round(pct / 100 * (len(s) - 1)))
    return s[idx]


def load_records(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    records = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def estimate_cost(record: dict, model: str) -> float:
    pricing = PRICING_PER_1K_TOKENS.get(model)
    if pricing is None:
        return 0.0
    tokens = record.get("tokens", {})
    prompt_cost = tokens.get("prompt", 0) / 1000 * pricing["prompt"]
    completion_cost = tokens.get("completion", 0) / 1000 * pricing["completion"]
    return prompt_cost + completion_cost


def is_declined(record: dict) -> bool:
    verdicts = record.get("critique_verdicts", [])
    return bool(verdicts) and verdicts[-1] == "fail (declined)"


def build_report(records: list[dict], model: str) -> dict:
    n = len(records)
    latencies = [r["total_latency_s"] for r in records]
    costs = [estimate_cost(r, model) for r in records]
    declined = [is_declined(r) for r in records]
    retried = [r.get("retried", False) for r in records]

    return {
        "n": n,
        "latency_p50_s": _percentile(latencies, 50),
        "latency_p95_s": _percentile(latencies, 95),
        "cost_per_request_mean": statistics.mean(costs) if costs else 0.0,
        "cost_total": sum(costs),
        "citation_coverage_pct": 100 * (1 - (sum(declined) / n)) if n else 0.0,
        "failure_rate_pct": 100 * (sum(declined) / n) if n else 0.0,
        "retry_rate_pct": 100 * (sum(retried) / n) if n else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate chat_trace.log into observability metrics.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--model", type=str, default=None, help="Model name for cost lookup (defaults to GEMINI_MODEL from config)")
    args = parser.parse_args()

    model = args.model
    if model is None:
        from app.config import GEMINI_MODEL
        model = GEMINI_MODEL

    records = load_records(args.log)
    if not records:
        print(f"No trace records found at {args.log}. Have some real chat turns happened yet?")
        return

    report = build_report(records, model)

    print(f"Trace log: {args.log}")
    print(f"Model (for cost lookup): {model}\n")
    print("--- summary (paste into METRICS.md) ---")
    print(f"Turns analyzed: {report['n']}")
    print(f"Latency: p50={report['latency_p50_s']:.2f}s  p95={report['latency_p95_s']:.2f}s")
    print(f"Cost per request: mean=${report['cost_per_request_mean']:.5f}  total=${report['cost_total']:.5f}")
    print(f"Citation coverage: {report['citation_coverage_pct']:.1f}%  (answers not declined)")
    print(f"Failure rate: {report['failure_rate_pct']:.1f}%  (declined -- no groundable answer found)")
    print(f"Retry rate: {report['retry_rate_pct']:.1f}%  (needed a self-critique retry)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\pyvenvs\p35backend\Scripts\python.exe -m pytest tests/test_trace_report.py -v` (from `backend/`)
Expected: all 9 tests PASS.

- [ ] **Step 5: Run the full backend test suite**

Run: `C:\pyvenvs\p35backend\Scripts\python.exe -m pytest -q` (from `backend/`)
Expected: all tests PASS (this is a purely additive new module — nothing else should be affected).

- [ ] **Step 6: Commit**

```bash
git add backend/app/trace_report.py backend/tests/test_trace_report.py
git commit -m "Add observability aggregation: P50/P95 latency, cost/request, citation coverage, failure rate"
```
