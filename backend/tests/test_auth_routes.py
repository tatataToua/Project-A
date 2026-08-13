"""Covers the OAuth routes in app/auth.py (login redirect, callback outcomes,
/auth/me, /auth/logout) with the Google client stubbed out -- no live calls to
accounts.google.com. `require_user` itself is covered in test_auth.py."""
import time
from types import SimpleNamespace

import httpx
import pytest
from authlib.integrations.base_client.errors import OAuthError
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse

from app import auth
from test_auth import SECRET, make_session_cookie

FRONTEND = "http://frontend.test"


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setattr(auth, "FRONTEND_URL", FRONTEND)
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key=SECRET, same_site="lax")
    app.include_router(auth.router)
    return TestClient(app, follow_redirects=False)


def stub_google(monkeypatch, *, token=None, error=None):
    """Replace the Authlib Google client with a stub that records the redirect
    URI it was handed and returns/raises whatever the test asks for."""
    calls = {}

    async def authorize_redirect(request, redirect_uri):
        calls["redirect_uri"] = redirect_uri
        return RedirectResponse("https://accounts.google.com/o/oauth2/v2/auth?stubbed=1")

    async def authorize_access_token(request):
        if error is not None:
            raise error
        return token

    monkeypatch.setattr(
        auth,
        "oauth",
        SimpleNamespace(
            google=SimpleNamespace(
                authorize_redirect=authorize_redirect,
                authorize_access_token=authorize_access_token,
            )
        ),
    )
    return calls


def test_login_redirects_to_google_with_callback_uri(client, monkeypatch):
    calls = stub_google(monkeypatch)

    resp = client.get("/auth/login")

    assert resp.status_code == 307
    assert resp.headers["location"].startswith("https://accounts.google.com/")
    assert str(calls["redirect_uri"]).endswith("/auth/callback")


def test_callback_stores_user_in_session_and_redirects_to_frontend(client, monkeypatch):
    stub_google(
        monkeypatch,
        token={"userinfo": {"email": "a@example.com", "name": "A"}},
    )
    resp = client.get("/auth/callback")

    assert resp.status_code == 307
    assert resp.headers["location"] == FRONTEND

    me = client.get("/auth/me")
    assert me.status_code == 200
    assert me.json() == {"email": "a@example.com", "name": "A"}


def test_callback_defaults_missing_name_to_empty_string(client, monkeypatch):
    stub_google(monkeypatch, token={"userinfo": {"email": "a@example.com"}})

    client.get("/auth/callback")

    assert client.get("/auth/me").json() == {"email": "a@example.com", "name": ""}


@pytest.mark.parametrize(
    "token",
    [{}, {"userinfo": None}, {"userinfo": {"name": "No Email"}}, {"userinfo": {"email": ""}}],
    ids=["no-userinfo-key", "null-userinfo", "userinfo-without-email", "empty-email"],
)
def test_callback_without_usable_userinfo_redirects_with_error(client, monkeypatch, token):
    stub_google(monkeypatch, token=token)

    resp = client.get("/auth/callback")

    assert resp.headers["location"] == f"{FRONTEND}/?error=login_failed"
    assert client.get("/auth/me").status_code == 401


def test_callback_oauth_error_redirects_with_error(client, monkeypatch):
    stub_google(monkeypatch, error=OAuthError(error="access_denied"))

    resp = client.get("/auth/callback")

    assert resp.status_code == 307
    assert resp.headers["location"] == f"{FRONTEND}/?error=login_failed"


def test_callback_unreachable_google_returns_502(client, monkeypatch):
    stub_google(monkeypatch, error=httpx.ConnectError("boom"))

    resp = client.get("/auth/callback")

    assert resp.status_code == 502
    assert resp.json()["detail"] == "Could not reach Google"


def test_me_rejects_missing_session(client):
    assert client.get("/auth/me").status_code == 401


def test_me_rejects_expired_session(client):
    client.cookies.set(
        "session", make_session_cookie({"email": "a@example.com", "name": "A", "exp": time.time() - 1})
    )
    assert client.get("/auth/me").status_code == 401


def test_logout_clears_the_session(client):
    client.cookies.set(
        "session",
        make_session_cookie({"email": "a@example.com", "name": "A", "exp": time.time() + 3600}),
    )

    resp = client.post("/auth/logout")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    client.cookies.clear()
    assert client.get("/auth/me").status_code == 401
