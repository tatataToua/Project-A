import time

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app import ratelimit
from app.auth import require_user

FAKE_USER = {"email": "limit@example.com", "name": "Limit", "exp": time.time() + 3600}


def make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/protected")
    def protected(_: None = Depends(ratelimit.enforce_chat_rate_limit)):
        return {"ok": True}

    app.dependency_overrides[require_user] = lambda: FAKE_USER
    return app


def test_allows_requests_under_the_limit(monkeypatch):
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_MAX_REQUESTS", 3)
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_WINDOW_SECONDS", 60)
    ratelimit._requests_by_user.clear()
    client = TestClient(make_app())

    for _ in range(3):
        assert client.get("/protected").status_code == 200


def test_blocks_requests_over_the_limit(monkeypatch):
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_MAX_REQUESTS", 3)
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_WINDOW_SECONDS", 60)
    ratelimit._requests_by_user.clear()
    client = TestClient(make_app())

    for _ in range(3):
        client.get("/protected")
    resp = client.get("/protected")
    assert resp.status_code == 429


def test_window_resets_after_expiry(monkeypatch):
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_MAX_REQUESTS", 1)
    monkeypatch.setattr(ratelimit, "RATE_LIMIT_WINDOW_SECONDS", 0.05)
    ratelimit._requests_by_user.clear()
    client = TestClient(make_app())

    assert client.get("/protected").status_code == 200
    assert client.get("/protected").status_code == 429
    time.sleep(0.1)
    assert client.get("/protected").status_code == 200
