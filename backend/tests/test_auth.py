import base64
import json
import time

import itsdangerous
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from app.auth import require_user

SECRET = "test-secret"


def make_app() -> FastAPI:
    """A minimal app wired with require_user, isolated from the real app/config
    (no GOOGLE_CLIENT_ID etc. needed, no live Google calls)."""
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key=SECRET, same_site="lax")

    @app.get("/protected")
    def protected(user: dict = Depends(require_user)):
        return {"email": user["email"]}

    return app


def make_session_cookie(payload: dict, secret: str = SECRET) -> str:
    """Build a session cookie value the same way Starlette's SessionMiddleware
    would, so require_user can be exercised without a real login handshake."""
    data = base64.b64encode(json.dumps({"user": payload}).encode("utf-8"))
    return itsdangerous.TimestampSigner(secret).sign(data).decode("utf-8")


def test_require_user_rejects_missing_session():
    client = TestClient(make_app())
    resp = client.get("/protected")
    assert resp.status_code == 401


def test_require_user_accepts_valid_session():
    client = TestClient(make_app())
    cookie = make_session_cookie(
        {"email": "a@example.com", "name": "A", "exp": time.time() + 3600}
    )
    client.cookies.set("session", cookie)
    resp = client.get("/protected")
    assert resp.status_code == 200
    assert resp.json() == {"email": "a@example.com"}


def test_require_user_rejects_expired_session():
    client = TestClient(make_app())
    cookie = make_session_cookie(
        {"email": "a@example.com", "name": "A", "exp": time.time() - 10}
    )
    client.cookies.set("session", cookie)
    resp = client.get("/protected")
    assert resp.status_code == 401
