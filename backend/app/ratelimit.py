"""In-memory per-user rate limiting for /chat.

Deliberately in-memory, not Redis-backed -- fine at today's single-process,
low-traffic scale, and consistent with the rest of the app not having a
Redis dependency yet (see ROADMAP.md Phase 5). Swapping _requests_by_user
for a Redis-backed store later is a drop-in replacement, not a rewrite of
the endpoints that use this dependency.
"""
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException

from app.auth import require_user
from app.config import RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS

_requests_by_user: dict[str, deque] = defaultdict(deque)


def enforce_chat_rate_limit(user: dict = Depends(require_user)) -> None:
    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    timestamps = _requests_by_user[user["email"]]

    while timestamps and timestamps[0] < window_start:
        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests -- limit is {RATE_LIMIT_MAX_REQUESTS} per {RATE_LIMIT_WINDOW_SECONDS}s.",
        )

    timestamps.append(now)
