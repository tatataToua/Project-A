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
import types

# --- Compatibility shim -----------------------------------------------------
# ragas 0.4.3 unconditionally does
#   `from langchain_community.chat_models.vertexai import ChatVertexAI`
# at import time (ragas/llms/base.py), just to list it in a static
# isinstance() allowlist. langchain-community >=0.4 dropped that submodule
# entirely (Vertex AI chat support now lives in the separate
# langchain-google-vertexai package), so the bare `import ragas` crashes with
# ModuleNotFoundError on any currently-installable langchain-community. We
# never use Vertex AI here, so we install a harmless stand-in module before
# ragas imports it, rather than pinning langchain-community to an old release
# (which would drag in an incompatible langchain-core and its own conflicts).
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # placeholder; never instantiated by this script
        pass

    _vertexai_stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub
# --- end compatibility shim -------------------------------------------------

import math

import yaml
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import RunConfig, evaluate
from ragas.metrics import faithfulness

from app.config import CONTENT_DIR, GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_MODEL
from app.db import SessionLocal
from app.embeddings import embed_texts
from app.models import Tenant
from app.retrieval import retrieve_chunks
from app.workflow import run_chat_workflow

FAITHFULNESS_THRESHOLD = 0.7

# A small local judge model occasionally returns JSON that doesn't match
# Ragas's expected statement/verdict schema for one item (OutputParserException),
# independent of the overall retrieval+generation quality being measured --
# observed: 1/30 on a real run, judge model qwen2.5:7b-instruct, mean 0.914
# over the other 29. That's noise, not a regression signal, so a handful of
# such flakes shouldn't redline the whole gate. A run where most/all items
# fail to score (e.g. the judge is unreachable, or overwhelmed -- concretely
# hit here as 30/30 TimeoutErrors before RunConfig concurrency was tuned for
# a local judge) is a real infrastructure failure and must still fail loudly
# rather than silently averaging over zero successes.
MAX_FAILED_SCORE_FRACTION = 0.1


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

        # golden_qa.yaml is human-authored YAML; a bare scalar like
        # `expected_answer: 2015.` parses as a Python float, not a string
        # (YAML doesn't know it's meant as prose). pyarrow then rejects the
        # column when it can't unify str and float entries. We don't edit
        # golden_qa.yaml (it's verified content), so coerce defensively here.
        questions.append(str(question))
        answers.append(str(answer))
        contexts.append([str(chunk_text) for _source, chunk_text in retrieved])
        references.append(str(item["expected_answer"]))

    # Column names here are Ragas's "v1" schema (question/answer/contexts).
    # The installed ragas (0.4.3) auto-renames these to the current v2 schema
    # (user_input/response/retrieved_contexts) inside evaluate() via
    # ragas.utils.convert_v1_to_v2_dataset -- confirmed by reading
    # ragas/evaluation.py locally. `reference` is already the v2 name and is
    # passed through unchanged. No manual renaming needed.
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

    # Ragas's default RunConfig (max_workers=16) dispatches up to 16 concurrent
    # judge calls -- fine against a hosted API, but a local Ollama server
    # effectively serializes generation on one model instance, so that
    # concurrency just queues requests behind each other. Worse, Ragas retries
    # timeouts by resubmitting into the same saturated queue, compounding the
    # backlog. Measured against this machine's Ollama instance: concurrent
    # requests each add ~1.5-2s of queue wait, so max_workers=16 pushed
    # per-item wall time past the default 180s timeout for every single item
    # (confirmed: a run with defaults produced 30/30 TimeoutErrors and a NaN
    # mean). Serializing (max_workers=1) with a generous per-item timeout
    # avoids the pileup; a small max_retries keeps genuine failures from
    # silently retrying forever.
    run_config = RunConfig(timeout=300, max_workers=1, max_retries=2, max_wait=15)

    print("Scoring with Ragas (faithfulness) -- serialized against the local judge, this takes a while...")
    result = evaluate(dataset, metrics=[faithfulness], llm=judge_llm, run_config=run_config)
    scores = result.to_pandas()

    failed = scores["faithfulness"].isna()
    num_failed = int(failed.sum())
    num_scored = len(scores) - num_failed
    mean_faithfulness = scores["faithfulness"].mean()

    print(f"\n--- summary (paste into METRICS.md) ---")
    print(f"Tenant: {args.tenant}  Questions: {len(golden_set)}  Judge model: {GEMINI_MODEL}")
    print(f"Scored: {num_scored}/{len(golden_set)}  (failed to score: {num_failed})")
    print(f"Mean faithfulness: {mean_faithfulness:.3f}  (threshold: {FAITHFULNESS_THRESHOLD})")

    # pandas .mean() skips NaN by default, so a run where every judge call
    # failed would otherwise report math.nan, which silently passes
    # `nan < threshold` (always False in Python) -- that must fail the gate,
    # not pass it. A *few* failed items are tolerated (see
    # MAX_FAILED_SCORE_FRACTION above); beyond that fraction, treat it the
    # same as a total failure -- something is systemically wrong with the
    # judge/pipeline, not just an isolated small-model formatting flake.
    failed_fraction = num_failed / len(golden_set)
    if math.isnan(mean_faithfulness) or failed_fraction > MAX_FAILED_SCORE_FRACTION:
        print(f"\nFAILED: {num_failed}/{len(golden_set)} questions could not be scored by the judge "
              f"(timeouts or errors) -- {failed_fraction:.0%} exceeds the "
              f"{MAX_FAILED_SCORE_FRACTION:.0%} tolerance for isolated judge flakes, so this run is "
              "treated as an infrastructure failure rather than averaged over the successful subset.")
        sys.exit(1)

    if num_failed > 0:
        print(f"\nNote: {num_failed}/{len(golden_set)} questions failed to score (within the "
              f"{MAX_FAILED_SCORE_FRACTION:.0%} tolerance) -- mean is over the remaining {num_scored}.")

    if mean_faithfulness < FAITHFULNESS_THRESHOLD:
        print(f"\nFAILED: mean faithfulness {mean_faithfulness:.3f} is below threshold {FAITHFULNESS_THRESHOLD}")
        sys.exit(1)

    print("\nPASSED: faithfulness above threshold.")


if __name__ == "__main__":
    main()
