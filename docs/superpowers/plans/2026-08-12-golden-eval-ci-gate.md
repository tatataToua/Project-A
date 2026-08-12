# Golden Eval Set + Ragas + CI Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score the (already-drafted, human-verified) golden Q&A set against the real chat workflow using Ragas faithfulness, and gate CI on it — the first CI workflow in this repo. This is sub-project #4 of the Phase 3.5 design spec.

**Architecture:** `docs/content/tenants/two-owls-tavern/eval/golden_qa.yaml` (already created — 30 Q&A pairs drafted from the tenant's actual content) is scored by a new `app/eval_rag.py`, which runs each question through the real `retrieve_chunks` + `run_chat_workflow`, then hands the (question, answer, retrieved contexts, reference answer) tuples to Ragas's `faithfulness` metric, using a `langchain_openai.ChatOpenAI` judge pointed at whatever provider `.env` already configures (Gemini or Ollama — same dual-provider story this repo already tells). `.github/workflows/eval.yml` runs this in CI against **Ollama**, not a hosted API — no GitHub secret needed, keeps the $0-cost story intact, and matches this repo's already-documented Ollama verification. The multi-GB model pull is cached via `actions/cache` so only the first CI run pays that cost.

**Tech Stack:** `ragas`, `langchain_openai`, `datasets` (Ragas's own dependency), `pyyaml` — 4 new dependencies.

**Scoped down from the original design spec for a first working version:** only `faithfulness` is scored (not context_precision/context_recall/answer_relevancy) — those need additional embedding-model wiring and ground-truth-context matching that add real integration risk without changing the core "does CI catch regressions" story. Documented as a fast-follow, not silently dropped.

## Global Constraints

- Windows dev environment: run backend commands via `C:\pyvenvs\p35backend\Scripts\python.exe`. Note: `ragas`/`langchain_openai`/`datasets` need to be installed into that venv before Task 1 can run for real (`C:\pyvenvs\p35backend\Scripts\python.exe -m pip install ragas langchain_openai datasets pyyaml`).
- The golden set (`docs/content/tenants/two-owls-tavern/eval/golden_qa.yaml`) already exists — do not regenerate or edit its content in this plan; if you find a factual error in it, flag it in your report rather than silently fixing it (a human already verified it against `about.md`/`faq.md`/`menu.md`).
- CI must not require any GitHub secret — Ollama only, by explicit decision (avoids exposing an API key, keeps $0 cost).
- This is genuinely new third-party API surface (Ragas) this session hasn't run before. If the exact `evaluate()`/`Dataset` column-naming convention in the code below doesn't match what the installed `ragas` version expects, adapt the column names/API calls to match — check the installed package's actual signature (`python -c "import ragas; help(ragas.evaluate)"` or read `ragas/__init__.py` in site-packages) rather than guessing blindly. This is expected adaptive work, not a sign something is wrong with the plan.

---

## Task 1: `app/eval_rag.py` — golden-set scoring script

**Files:**
- Modify: `backend/requirements.txt`
- Create: `backend/app/eval_rag.py`

**Interfaces:**
- Standalone CLI (`python -m app.eval_rag --tenant <slug>`), exits non-zero if mean
  faithfulness drops below `FAITHFULNESS_THRESHOLD`. Consumed by Task 2's CI workflow.

- [x] **Step 1: Add dependencies and install them**

In `backend/requirements.txt`, add after the `sentence-transformers` line:

```
ragas>=0.2
langchain_openai>=0.2
datasets>=3.0
pyyaml>=6.0
```

Run: `C:\pyvenvs\p35backend\Scripts\python.exe -m pip install -r backend\requirements.txt` (from the repo root, or drop the `backend\` prefix if run from `backend/`)

This is a real, possibly slow install (Ragas pulls in a fair dependency tree). Be patient.

- [x] **Step 2: Implement `app/eval_rag.py`**

Result: implemented as in the snippet below, plus a `langchain_community.chat_models.vertexai`
compatibility shim (ragas 0.4.3 unconditionally imports it; current langchain-community dropped
it) and `str()` coercion on golden-set fields that YAML parses as non-string scalars (e.g.
`expected_answer: 2015.` → Python `float`, which pyarrow then rejects mixed with `str` entries).

```python
"""Golden-set evaluation: runs each golden Q&A pair through the real chat
workflow, scores faithfulness with Ragas, and exits non-zero if the mean
score drops below FAITHFULNESS_THRESHOLD. Wired into CI
(.github/workflows/eval.yml) as a build-blocking gate.

Requires the same environment as running the app: `backend/.env` configured,
Postgres up, and the tenant already ingested (`python -m app.ingest <slug>`).

Usage (from `backend/`):
    .venv\\Scripts\\python.exe -m app.eval_rag --tenant two-owls-tavern
"""
import argparse
import sys

import yaml
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import faithfulness

from app.config import CONTENT_DIR, GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_MODEL
from app.db import SessionLocal
from app.embeddings import embed_texts
from app.models import Tenant
from app.retrieval import retrieve_chunks
from app.workflow import run_chat_workflow

FAITHFULNESS_THRESHOLD = 0.7


def load_golden_set(tenant_slug: str) -> list[dict]:
    path = CONTENT_DIR / "tenants" / tenant_slug / "eval" / "golden_qa.yaml"
    if not path.exists():
        raise SystemExit(f"No golden eval set at {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_dataset(tenant_slug: str, golden_set: list[dict]) -> Dataset:
    session = SessionLocal()
    try:
        tenant = session.query(Tenant).filter_by(slug=tenant_slug).one()
    finally:
        session.close()

    questions, answers, contexts, references = [], [], [], []
    for i, item in enumerate(golden_set, start=1):
        question = item["question"]
        print(f"  [{i}/{len(golden_set)}] {question}")
        [query_vector] = embed_texts([question])
        retrieved = retrieve_chunks(tenant.id, question, query_vector)
        answer = run_chat_workflow(tenant.id, question)

        questions.append(question)
        answers.append(answer)
        contexts.append([chunk_text for _source, chunk_text in retrieved])
        references.append(item["expected_answer"])

    return Dataset.from_dict(
        {"question": questions, "answer": answers, "contexts": contexts, "reference": references}
    )


def main():
    parser = argparse.ArgumentParser(description="Score the tenant's golden eval set with Ragas.")
    parser.add_argument("--tenant", default="two-owls-tavern")
    args = parser.parse_args()

    golden_set = load_golden_set(args.tenant)
    print(f"Loaded {len(golden_set)} golden Q&A pairs for tenant '{args.tenant}'")

    print("Running each question through the real chat workflow (this calls the live LLM)...")
    dataset = build_dataset(args.tenant, golden_set)

    judge_llm = ChatOpenAI(base_url=GEMINI_BASE_URL, api_key=GEMINI_API_KEY or "unused", model=GEMINI_MODEL)

    print("Scoring with Ragas (faithfulness)...")
    result = evaluate(dataset, metrics=[faithfulness], llm=judge_llm)
    scores = result.to_pandas()

    mean_faithfulness = scores["faithfulness"].mean()

    print(f"\n--- summary (paste into METRICS.md) ---")
    print(f"Tenant: {args.tenant}  Questions: {len(golden_set)}  Judge model: {GEMINI_MODEL}")
    print(f"Mean faithfulness: {mean_faithfulness:.3f}  (threshold: {FAITHFULNESS_THRESHOLD})")

    if mean_faithfulness < FAITHFULNESS_THRESHOLD:
        print(f"\nFAILED: mean faithfulness {mean_faithfulness:.3f} is below threshold {FAITHFULNESS_THRESHOLD}")
        sys.exit(1)

    print("\nPASSED: faithfulness above threshold.")


if __name__ == "__main__":
    main()
```

If the installed `ragas` version's `evaluate()` or `Dataset.from_dict()` expects different
column names than `question`/`answer`/`contexts`/`reference` (Ragas has renamed these
across versions — some versions use `user_input`/`response`/`retrieved_contexts`), adapt
the dict keys in `build_dataset` and the `scores["faithfulness"]` lookup to match. Check
with `python -c "import ragas; print(ragas.__version__)"` and the installed package's
own docs/docstrings.

- [x] **Step 3: Run it for real against the local Ollama provider**

Confirm `backend/.env` has `GEMINI_BASE_URL=http://localhost:11434/v1` and
`GEMINI_MODEL=qwen2.5:7b-instruct` set (it already does per this session's setup), Postgres
is running, and the tenant is ingested (`C:\pyvenvs\p35backend\Scripts\python.exe -m app.ingest two-owls-tavern` from `backend/`).

Run: `C:\pyvenvs\p35backend\Scripts\python.exe -m app.eval_rag --tenant two-owls-tavern` (from `backend/`)

Result: the first real run surfaced two genuine bugs, not just an "adapt column names" mismatch:

1. **Silent-pass-on-total-failure bug.** Ragas's default `RunConfig` dispatches up to 16
   concurrent judge calls. Fine against a hosted API, but this local Ollama instance
   effectively serializes generation on one model — concurrent requests just queue behind
   each other (measured: ~1.5-2s added per concurrent request), and Ragas's retry-on-timeout
   behavior resubmits into the same already-saturated queue, compounding the backlog. Result:
   30/30 `TimeoutError`, and because `pandas.Series.mean()` skips `NaN` and `nan < threshold`
   is always `False` in Python, the script printed **"PASSED"** with a `NaN` mean and zero
   real scores — a build-blocking gate that can never actually block. Fixed by serializing
   judge calls (`RunConfig(max_workers=1, timeout=300)`) and hard-failing whenever the
   failed-score fraction exceeds `MAX_FAILED_SCORE_FRACTION` (10%), instead of silently
   averaging over whichever subset happened to succeed.
2. **Isolated small-model formatting flakes.** Even serialized, the local 7B judge
   occasionally drops the last field when asked to enumerate a longer list of statement
   verdicts (`OutputParserException`, Pydantic "Field required"). Two separate full runs each
   hit exactly one such flake, on different questions — confirmed as judge-side noise, not a
   retrieval/generation regression, since the rest of each run scored consistently high. Given
   a tolerance (implemented above), this doesn't fail the gate.

Final passing run: **28/30 scored, mean faithfulness 0.948** (threshold 0.7). Judge model
`qwen2.5:7b-instruct`. Paste-ready for `METRICS.md`.

- [x] **Step 4: Run the full backend test suite** (confirms the new dependencies didn't
  break anything else)

Run: `C:\pyvenvs\p35backend\Scripts\python.exe -m pytest -q` (from `backend/`)
Result: 60/60 passed.

- [x] **Step 5: Commit**

```bash
git add backend/requirements.txt backend/app/eval_rag.py
git commit -m "Add Ragas faithfulness scoring for the golden eval set"
```

Result: committed as `bf5c45e`.

---

## Task 2: CI workflow — eval gate on Ollama

**Files:**
- Create: `.github/workflows/eval.yml`

**Interfaces:**
- Standalone GitHub Actions workflow, invokes Task 1's `app.eval_rag` CLI.

- [x] **Step 1: Implement `.github/workflows/eval.yml`**

```yaml
name: RAG Eval Gate

on:
  pull_request:
    paths:
      - 'backend/**'
      - 'docs/content/**'

jobs:
  eval:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_USER: askme
          POSTGRES_PASSWORD: askme
          POSTGRES_DB: askme
        ports:
          - 5432:5432
        options: >-
          --health-cmd "pg_isready -U askme"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    env:
      DATABASE_URL: postgresql+psycopg://askme:askme@localhost:5432/askme
      GEMINI_BASE_URL: http://localhost:11434/v1
      GEMINI_MODEL: qwen2.5:7b-instruct
      GEMINI_EMBEDDING_MODEL: nomic-embed-text
      GEMINI_API_KEY: unused

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Cache pip packages
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: pip-${{ hashFiles('backend/requirements.txt') }}

      - name: Install backend dependencies
        run: pip install -r backend/requirements.txt

      - name: Install Ollama
        run: curl -fsSL https://ollama.com/install.sh | sh

      - name: Cache Ollama models
        uses: actions/cache@v4
        with:
          path: ~/.ollama/models
          key: ollama-models-qwen2.5-7b-instruct-nomic-embed-text

      - name: Start Ollama and pull models
        run: |
          ollama serve &
          sleep 5
          ollama pull qwen2.5:7b-instruct
          ollama pull nomic-embed-text

      - name: Ingest eval tenant content
        working-directory: backend
        run: python -m app.ingest two-owls-tavern

      - name: Run golden eval set (Ragas faithfulness gate)
        working-directory: backend
        run: python -m app.eval_rag --tenant two-owls-tavern
```

- [x] **Step 2: Validate the YAML structurally**

Result: YAML parses cleanly (`yaml.safe_load`), and the final step's command
(`python -m app.eval_rag --tenant two-owls-tavern`, `working-directory: backend`) matches
`app/eval_rag.py`'s actual CLI. This workflow's first real run only happens once it's pushed
to GitHub — not verified from this local checkout, as expected.

- [x] **Step 3: Commit**

```bash
git add .github/workflows/eval.yml
git commit -m "Add CI eval gate (Ragas faithfulness, against Ollama, no secret required)"
```

Result: committed as `5a59973`.
