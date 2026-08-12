# Local-Model Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reusable script that benchmarks 3 local Ollama models sequentially (tokens/sec, time-to-first-token) against a fixed prompt set, matching `bench_chat.py`'s existing "run and paste into METRICS.md" pattern. This is the "local-model benchmark" half of sub-project #6 of the Phase 3.5 design spec (the other half, classify-node hardening, is a separate plan since it touches `workflow.py`, which citation-enforcement is also modifying).

**Architecture:** `backend/scripts/bench_models.py`, matching `bench_chat.py`'s existing location and style (a runnable harness, not unit-tested — this repo's established convention: `scripts/` holds performance/benchmark tools exercised by running them, `app/` holds logic worth unit-testing). Builds its own `OpenAI` client pointed at Ollama's OpenAI-compatible endpoint per model (bypassing the app's cached `app.llm.client`, which is fixed to whatever `GEMINI_MODEL`/`GEMINI_BASE_URL` are in `.env` at import time) so it can swap models between iterations without restarting the process. Models run **strictly sequentially** — only one model is ever loaded in Ollama's memory/VRAM at a time.

**Tech Stack:** `openai` SDK (already a dependency) pointed at `http://localhost:11434/v1`, streaming responses to measure time-to-first-token.

## Global Constraints

- No new Python dependencies.
- Windows dev environment: run backend commands via `C:\pyvenvs\p35backend\Scripts\python.exe`.
- Requires Ollama running locally (`ollama serve`, already running per earlier verification in this session) — the script pulls any of the 3 models not already present, which costs real time/bandwidth (multi-GB downloads) the first time.
- Models: `qwen2.5:7b-instruct` (already pulled, used elsewhere in this repo), `llama3.2`, `mistral:7b` (both need pulling).

---

## Task 1: `scripts/bench_models.py`

**Files:**
- Create: `backend/scripts/bench_models.py`

**Interfaces:**
- Standalone script, no importable interface consumed elsewhere — matches `bench_chat.py`.

- [ ] **Step 1: Implement the script**

Create `backend/scripts/bench_models.py`:

```python
"""Sequential local-model benchmark: mean latency, time-to-first-token, and
tokens/sec across a fixed set of Ollama models, run strictly one at a time --
only one model is ever loaded in Ollama's memory/VRAM at once. Paste the
printed summary table into METRICS.md.

Requires Ollama running locally (`ollama serve`). Pulls any model in MODELS
that isn't already present -- the first run may download several GB.

Usage (from `backend/`):
    .venv\\Scripts\\python.exe scripts\\bench_models.py
"""
import statistics
import subprocess
import time

from openai import OpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODELS = ["qwen2.5:7b-instruct", "llama3.2", "mistral:7b"]

PROMPTS = [
    "What are your hours on Saturday?",
    "Do you have vegan options on the menu?",
    "What's the story behind the name Two Owls Tavern?",
    "Can I book a table for 10 people?",
    "What's your most popular dish?",
]


def ensure_model_pulled(model: str) -> None:
    print(f"  Ensuring '{model}' is pulled...")
    subprocess.run(["ollama", "pull", model], check=True)


def benchmark_model(model: str) -> dict:
    client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
    latencies = []
    tokens_per_sec = []
    ttft_samples = []

    for prompt in PROMPTS:
        start = time.perf_counter()
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        first_token_time = None
        completion_tokens = 0
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                completion_tokens += 1  # rough proxy: one "token" per stream chunk
        end = time.perf_counter()

        total_time = end - start
        ttft = (first_token_time - start) if first_token_time else total_time
        latencies.append(total_time)
        ttft_samples.append(ttft)
        if total_time > 0 and completion_tokens > 0:
            tokens_per_sec.append(completion_tokens / total_time)

    return {
        "model": model,
        "mean_latency_s": statistics.mean(latencies),
        "mean_ttft_s": statistics.mean(ttft_samples),
        "mean_tokens_per_sec": statistics.mean(tokens_per_sec) if tokens_per_sec else 0.0,
    }


def main():
    print(f"Benchmarking {len(MODELS)} models sequentially against {len(PROMPTS)} prompts each.")
    print("Only one model is loaded in Ollama's memory at a time.\n")

    results = []
    for model in MODELS:
        print(f"=== {model} ===")
        ensure_model_pulled(model)
        result = benchmark_model(model)
        results.append(result)
        print(
            f"  latency={result['mean_latency_s']:.2f}s  "
            f"ttft={result['mean_ttft_s']:.2f}s  "
            f"tokens/sec={result['mean_tokens_per_sec']:.1f}\n"
        )

    print("--- summary table (paste into METRICS.md) ---")
    print(f"{'Model':<25} {'Mean latency (s)':>18} {'Mean TTFT (s)':>16} {'Tokens/sec':>12}")
    for r in results:
        print(f"{r['model']:<25} {r['mean_latency_s']:>18.2f} {r['mean_ttft_s']:>16.2f} {r['mean_tokens_per_sec']:>12.1f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it for real and verify sane output**

Run: `C:\pyvenvs\p35backend\Scripts\python.exe scripts\bench_models.py` (from `backend/`)

This is a real benchmark, not a unit test — there's no RED/GREEN cycle. Expected: the
script pulls `llama3.2` and `mistral:7b` (may take several minutes the first time),
then prints per-model progress and a final summary table with 3 rows, each with
non-zero latency/TTFT/tokens-per-sec numbers. If a model fails to pull or Ollama isn't
reachable, the script should error clearly (via the `subprocess.run(..., check=True)`
and the `OpenAI` client's own connection errors) rather than hang silently — confirm
this by checking Ollama is running first (`curl http://localhost:11434/api/tags`)
before troubleshooting further.

Capture the full console output (the per-model progress and the final summary table)
for your report — this is the evidence a human will use to paste into `METRICS.md`
later.

- [ ] **Step 3: Run the full backend test suite** (confirms nothing else broke — this
  task adds a new file only, doesn't modify existing code)

Run: `C:\pyvenvs\p35backend\Scripts\python.exe -m pytest -q` (from `backend/`)
Expected: all existing tests still PASS.

- [ ] **Step 4: Commit**

```bash
git add backend/scripts/bench_models.py
git commit -m "Add sequential local-model benchmark script (tokens/sec, TTFT)"
```
