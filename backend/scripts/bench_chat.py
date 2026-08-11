"""Reusable benchmark for METRICS.md ledger entries.

Runs the test suite, then sends a fixed set of sample questions through
`run_chat_workflow` against the live `two-owls-tavern` tenant, measuring
wall-clock latency and token usage per query. Prints a summary formatted
for pasting straight into a new METRICS.md ledger row.

Requires the same environment as running the app: `backend/.env` configured,
Postgres up (`docker compose up -d`), and the tenant already ingested
(`python -m app.ingest two-owls-tavern`).

Usage (from `backend/`):
    .venv\\Scripts\\python.exe scripts\\bench_chat.py
"""
import os
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import GEMINI_BASE_URL, GEMINI_EMBEDDING_MODEL, GEMINI_MODEL
from app.db import SessionLocal
from app.models import Tenant
from app import llm as llm_module
from app.workflow import run_chat_workflow

QUESTIONS = [
    "What are your hours on Saturday?",
    "Do you have vegan options on the menu?",
    "What's the story behind the name Two Owls Tavern?",
    "Can I book a table for 10 people?",
    "What's your most popular dish?",
]

TENANT_SLUG = "two-owls-tavern"


def run_tests() -> tuple[int, float]:
    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start
    summary_line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    print(summary_line or "(no test output)")
    match = re.search(r"(\d+) passed", summary_line)
    passed = int(match.group(1)) if match else 0
    return passed, elapsed


def run_chat_benchmark() -> list[dict]:
    usage_log = []
    original_create = llm_module.client.chat.completions.create

    def tracking_create(*args, **kwargs):
        response = original_create(*args, **kwargs)
        if response.usage:
            usage_log.append(
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            )
        return response

    llm_module.client.chat.completions.create = tracking_create

    db = SessionLocal()
    tenant = db.query(Tenant).filter(Tenant.slug == TENANT_SLUG).first()
    db.close()
    if tenant is None:
        raise SystemExit(
            f"Tenant '{TENANT_SLUG}' not found — run "
            f"`python -m app.ingest {TENANT_SLUG}` first."
        )

    results = []
    for question in QUESTIONS:
        usage_log.clear()
        start = time.perf_counter()
        answer = run_chat_workflow(tenant.id, question)
        elapsed = time.perf_counter() - start
        results.append(
            {
                "question": question,
                "elapsed_s": elapsed,
                "llm_calls": len(usage_log),
                "total_tokens": sum(u["total_tokens"] for u in usage_log),
            }
        )
        print(f"  [{elapsed:5.2f}s] calls={len(usage_log)} tokens={results[-1]['total_tokens']:>5}  {question}")
        _ = answer  # not asserted on -- this is a perf/cost benchmark, not a quality eval

    return results


def main():
    print(f"Provider: {GEMINI_BASE_URL}  model={GEMINI_MODEL}  embedding={GEMINI_EMBEDDING_MODEL}\n")

    print("Running test suite...")
    passed, test_elapsed = run_tests()
    print(f"  {passed} passed in {test_elapsed:.2f}s\n")

    print(f"Running {len(QUESTIONS)} sample chat queries against tenant '{TENANT_SLUG}'...")
    results = run_chat_benchmark()

    latencies = [r["elapsed_s"] for r in results]
    tokens = [r["total_tokens"] for r in results]
    retries = sum(1 for r in results if r["llm_calls"] > 3)

    print("\n--- summary (paste into METRICS.md) ---")
    print(f"Date: {time.strftime('%Y-%m-%d')}")
    print(f"Provider/model: {GEMINI_MODEL} (base_url={GEMINI_BASE_URL})")
    print(f"Tests: {passed} passed, {test_elapsed:.2f}s")
    print(
        f"Latency (n={len(results)}): mean={statistics.mean(latencies):.2f}s "
        f"median={statistics.median(latencies):.2f}s "
        f"min={min(latencies):.2f}s max={max(latencies):.2f}s"
    )
    print(f"Tokens: mean={statistics.mean(tokens):.0f} min={min(tokens)} max={max(tokens)}")
    print(f"Self-critique retries: {retries}/{len(results)}")


if __name__ == "__main__":
    main()
