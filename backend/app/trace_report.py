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
