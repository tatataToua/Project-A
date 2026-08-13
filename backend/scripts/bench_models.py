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

from bench_common import SAMPLE_QUESTIONS as PROMPTS

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODELS = ["qwen2.5:7b-instruct", "llama3.2", "mistral:7b"]


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
