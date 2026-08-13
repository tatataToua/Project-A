"""Tools for watching the chat workflow: timing, tokens, and critique verdicts
per turn -- for debugging and learning what classify -> retrieve -> generate
-> critique is doing.

Real chat turns from the browser widget (via `/chat/{tenant_slug}`) auto-log
to `backend/logs/chat_trace.log` on every request -- see app/tracing.py and
its use in app/main.py. This script is for *viewing* that log, plus an
interactive REPL for ad hoc questions outside the browser.

Usage (from backend/):
    python -m app.trace_chat --watch                # tail real (or REPL) turns live, run alongside the app
    python -m app.trace_chat --stats                 # aggregate stats across every logged turn
    python -m app.trace_chat --export-csv [path]     # export the full log to CSV for Excel/Sheets, full text, no truncation
    python -m app.trace_chat [tenant_slug]           # interactive REPL, prints its own live trace
"""
import argparse
import csv
import json
import time
from pathlib import Path

from app.db import SessionLocal
from app.models import Tenant
from app.tracing import LOG_PATH, instrument_client, read_records, trace_turn

CSV_FIELDNAMES = [
    "timestamp",
    "question",
    "node",
    "elapsed_s",
    "prompt_tokens",
    "completion_tokens",
    "text",
    "answer",
    "total_latency_s",
    "total_prompt_tokens",
    "total_completion_tokens",
    "first_pass",
    "retried",
]


def _format_node_line(node_name: str, elapsed: float, prompt_tok: int, completion_tok: int, preview: str) -> str:
    return f"  [{node_name:9s}] {elapsed:5.2f}s | {prompt_tok:4d} in / {completion_tok:3d} out tok | {preview}"


def _format_summary(record: dict) -> str:
    verdicts = ", ".join(record.get("critique_verdicts", []))
    status = "retried" if record.get("retried") else "first-pass"
    tokens = record.get("tokens", {})
    return (
        f"\n  answer: {record.get('answer', '')}\n\n"
        f"  total: {record.get('total_latency_s', 0):.2f}s | {tokens.get('prompt', 0)} in / "
        f"{tokens.get('completion', 0)} out tok | {status} | critique: {verdicts}\n"
    )


def _print_node(node_name: str, elapsed: float, prompt_tok: int, completion_tok: int, preview: str) -> None:
    print(_format_node_line(node_name, elapsed, prompt_tok, completion_tok, preview))


def _run_turn(tenant_id: int, question: str) -> None:
    _answer, record = trace_turn(tenant_id, question, on_node=_print_node)
    print(_format_summary(record))


def _format_logged_turn(record: dict) -> str:
    """Replay a completed turn in the same per-node format the REPL prints
    live, using the ordered event list captured by trace_turn."""
    lines = [f"You: {record.get('question', '')}"]
    for event in record.get("events", []):
        lines.append(
            _format_node_line(
                event["node"], event["elapsed_s"], event["prompt_tokens"], event["completion_tokens"], event["preview"]
            )
        )
    return "\n".join(lines) + "\n" + _format_summary(record)


def _watch() -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.touch()

    print(f"Watching {LOG_PATH} for new turns -- chat in the browser widget or another terminal. Ctrl+C to stop.\n")
    with LOG_PATH.open("r", encoding="utf-8") as f:
        f.seek(0, 2)  # start at end -- only show turns that happen from now on
        try:
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A turn still being appended, or a corrupt line -- keep watching.
                    print("  (skipped an unparseable trace line)")
                    continue
                print(_format_logged_turn(record))
        except KeyboardInterrupt:
            print()


def _export_csv(output_path: Path | None) -> None:
    if not LOG_PATH.exists():
        print(f"No trace log yet at {LOG_PATH} -- chat with the app or run a turn first.")
        return
    rows = read_records(LOG_PATH)
    if not rows:
        print("Trace log is empty.")
        return

    out_path = output_path or (LOG_PATH.parent / "chat_trace.csv")
    event_count = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for record in rows:
            for event in record.get("events", []):
                event_count += 1
                writer.writerow(
                    {
                        "timestamp": record["timestamp"],
                        "question": record["question"],
                        "node": event["node"],
                        "elapsed_s": event["elapsed_s"],
                        "prompt_tokens": event["prompt_tokens"],
                        "completion_tokens": event["completion_tokens"],
                        "text": event.get("full_text", event["preview"]),
                        "answer": record["answer"],
                        "total_latency_s": record["total_latency_s"],
                        "total_prompt_tokens": record["tokens"]["prompt"],
                        "total_completion_tokens": record["tokens"]["completion"],
                        "first_pass": record["first_pass"],
                        "retried": record["retried"],
                    }
                )
    print(f"Exported {len(rows)} turns ({event_count} node events) to {out_path}")


def _print_stats() -> None:
    if not LOG_PATH.exists():
        print(f"No trace log yet at {LOG_PATH} -- chat with the app or run a turn first.")
        return
    rows = read_records(LOG_PATH)
    if not rows:
        print("Trace log is empty.")
        return

    n = len(rows)
    first_pass_rate = sum(r["first_pass"] for r in rows) / n
    avg_total = sum(r["total_latency_s"] for r in rows) / n
    avg_tokens = sum(r["tokens"]["prompt"] + r["tokens"]["completion"] for r in rows) / n

    node_totals: dict[str, list[float]] = {}
    for r in rows:
        for node, secs in r["node_latency_s"].items():
            node_totals.setdefault(node, []).append(secs)

    print(f"Turns logged:      {n}")
    print(f"First-pass rate:   {first_pass_rate:.0%}  (answer passed critique without a retry)")
    print(f"Avg total latency: {avg_total:.2f}s")
    print(f"Avg tokens/turn:   {avg_tokens:.0f}")
    print("Avg latency by node:")
    for node, values in node_totals.items():
        print(f"  {node:10s} {sum(values) / len(values):.2f}s  (n={len(values)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tenant_slug", nargs="?", default="two-owls-tavern")
    parser.add_argument("--watch", action="store_true", help="Tail the trace log live as real turns land")
    parser.add_argument("--stats", action="store_true", help="Print aggregate stats from the trace log and exit")
    parser.add_argument(
        "--export-csv",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Export the full trace log to CSV (default: backend/logs/chat_trace.csv), full text, no truncation",
    )
    args = parser.parse_args()

    if args.export_csv is not None:
        _export_csv(Path(args.export_csv) if args.export_csv else None)
        return
    if args.stats:
        _print_stats()
        return
    if args.watch:
        _watch()
        return

    session = SessionLocal()
    try:
        tenant = session.query(Tenant).filter_by(slug=args.tenant_slug).one_or_none()
    finally:
        session.close()
    if tenant is None:
        print(f"No tenant '{args.tenant_slug}' found. Run `python -m app.ingest {args.tenant_slug}` first.")
        return

    instrument_client()

    print(f"Tracing chat workflow for tenant '{args.tenant_slug}'. Type a question, or 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        _run_turn(tenant.id, question)


if __name__ == "__main__":
    main()
