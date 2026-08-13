"""Live per-node tracing for the chat workflow: timing, token counts, and the
critique verdict for every turn -- streamed via LangGraph's `.stream()` instead
of the plain `.invoke()` the workflow itself uses elsewhere.

`trace_turn` is the single implementation, used two ways:
- `main.py`'s real `/chat/{tenant_slug}` endpoint calls it silently (no
  `on_node` callback) so every real browser conversation auto-logs to
  `chat_trace.log` with no extra typing.
- `trace_chat.py` passes a print callback for an interactive, live-printed
  REPL, and `--watch` tails the same log file from a second terminal.
"""
import functools
import json
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from app.trace_log import LOG_PATH
from app.workflow import _graph, initial_chat_state

_usage_buffer: list[dict] = []
_instrumented = False


def drain_usage(kind: str | None = None) -> list[dict]:
    """Take and clear the token usage recorded since the last drain. Each entry
    is {"kind": "chat"|"embedding", "prompt_tokens", "completion_tokens"};
    pass `kind` to keep only calls of that kind (the rest are still cleared)."""
    usage = list(_usage_buffer)
    _usage_buffer.clear()
    if kind is None:
        return usage
    return [u for u in usage if u["kind"] == kind]


def instrument_client() -> None:
    """Wrap the shared LLM client so every call records token usage, without
    touching app.llm/embeddings.py/workflow.py. Idempotent -- safe to call on
    every app startup."""
    global _instrumented
    if _instrumented:
        return

    from app.llm import client

    orig_chat_create = client.chat.completions.create
    orig_embed_create = client.embeddings.create

    def record_usage(kind: str, response, count_completion: bool) -> None:
        usage = getattr(response, "usage", None)
        _usage_buffer.append(
            {
                "kind": kind,
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": (getattr(usage, "completion_tokens", 0) or 0) if count_completion else 0,
            }
        )

    @functools.wraps(orig_chat_create)
    def chat_create(*args, **kwargs):
        completion = orig_chat_create(*args, **kwargs)
        record_usage("chat", completion, count_completion=True)
        return completion

    @functools.wraps(orig_embed_create)
    def embed_create(*args, **kwargs):
        response = orig_embed_create(*args, **kwargs)
        record_usage("embedding", response, count_completion=False)
        return response

    client.chat.completions.create = chat_create
    client.embeddings.create = embed_create
    _instrumented = True


def _preview_for_node(node_name: str, delta: dict) -> str:
    """Compact, truncated one-liner for live console printing."""
    if node_name == "classify":
        return f"category={delta.get('category')}"
    if node_name == "retrieve":
        return f"{len(delta.get('chunks', []))} chunks retrieved"
    if node_name == "generate":
        answer = delta.get("answer") or ""
        snippet = answer[:70] + ("..." if len(answer) > 70 else "")
        return f'answer="{snippet}"'
    if node_name == "critique":
        if delta.get("needs_retry"):
            return "fail (retrying)"
        if "answer" in delta:
            return "fail (declined)"
        return "pass"
    return str(delta)


def _full_text_for_node(node_name: str, delta: dict, preview: str) -> str:
    """Untruncated version of the preview, for storage/export. Only `generate`
    and `retrieve` actually differ from the console preview -- the others are
    already short."""
    if node_name == "generate":
        return delta.get("answer") or ""
    if node_name == "retrieve":
        return f'{preview} (embedded query: "{delta.get("search_text", "")}")'
    return preview


NodeCallback = Callable[[str, float, int, int, str], None]


def trace_turn(tenant_id: int, question: str, on_node: Optional[NodeCallback] = None) -> tuple[str, dict]:
    """Run one turn through the workflow graph, collecting per-node timing,
    token usage, and critique verdicts. Appends a JSONL record to LOG_PATH and
    returns (answer, record). If `on_node` is given, it's called after each
    node completes with (node_name, elapsed_seconds, prompt_tokens,
    completion_tokens, preview_text) -- for live printing."""
    state = initial_chat_state(tenant_id, question)

    turn_start = time.perf_counter()
    last_ts = turn_start
    node_latency: dict[str, float] = {}
    node_tokens: dict[str, dict[str, int]] = {}
    critique_verdicts: list[str] = []
    events: list[dict] = []
    final_state = dict(state)

    for update in _graph.stream(state, stream_mode="updates"):
        now = time.perf_counter()
        for node_name, delta in update.items():
            elapsed = now - last_ts
            last_ts = now

            tokens_this_node = drain_usage()
            prompt_tok = sum(u["prompt_tokens"] for u in tokens_this_node)
            completion_tok = sum(u["completion_tokens"] for u in tokens_this_node)

            final_state.update(delta)
            node_latency[node_name] = node_latency.get(node_name, 0.0) + elapsed
            totals = node_tokens.setdefault(node_name, {"prompt": 0, "completion": 0})
            totals["prompt"] += prompt_tok
            totals["completion"] += completion_tok

            preview = _preview_for_node(node_name, delta)
            if node_name == "critique":
                critique_verdicts.append(preview)
            events.append(
                {
                    "node": node_name,
                    "elapsed_s": round(elapsed, 3),
                    "prompt_tokens": prompt_tok,
                    "completion_tokens": completion_tok,
                    "preview": preview,
                    "full_text": _full_text_for_node(node_name, delta, preview),
                }
            )
            if on_node is not None:
                on_node(node_name, elapsed, prompt_tok, completion_tok, preview)

    total_latency = time.perf_counter() - turn_start
    total_tokens = {
        "prompt": sum(v["prompt"] for v in node_tokens.values()),
        "completion": sum(v["completion"] for v in node_tokens.values()),
    }
    first_pass = bool(critique_verdicts) and critique_verdicts[0] == "pass"
    retried = final_state.get("retry_used", False)
    answer = final_state.get("answer", "")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
        "total_latency_s": round(total_latency, 3),
        "node_latency_s": {k: round(v, 3) for k, v in node_latency.items()},
        "tokens": total_tokens,
        "node_tokens": node_tokens,
        "first_pass": first_pass,
        "retried": retried,
        "critique_verdicts": critique_verdicts,
        "events": events,
    }

    LOG_PATH.parent.mkdir(exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return answer, record
