"""The chat trace log: where it lives and how to read it.

`tracing.py` appends one JSON record per turn here; `trace_chat.py` (watch,
stats, CSV export) and `trace_report.py` (aggregate metrics) read it back.
Both the path and the JSONL parsing live here so those readers can't drift
apart from the writer.
"""
import json
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "chat_trace.log"


def load_records(log_path: Path = LOG_PATH) -> list[dict]:
    """Read a JSONL trace log into records, ignoring blank lines. Returns an
    empty list if the log doesn't exist yet."""
    if not log_path.exists():
        return []
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
