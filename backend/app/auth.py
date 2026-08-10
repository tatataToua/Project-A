"""Google OAuth login, session-backed auth, and the require_user dependency.

The Starlette session cookie (configured in main.py) does double duty:
- during the login handshake: OAuth state/nonce/PKCE (managed by Authlib)
- after login: {"user": {"email": ..., "name": ..., "exp": <unix timestamp>}}
"""
import time

import httpx
from authlib.integrations.base_client.errors import OAuthError
from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from app.config import (
    FRONTEND_URL,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    SESSION_MAX_AGE_SECONDS,
)

router = APIRouter()

oauth = OAuth()
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
    code_challenge_method="S256",  # PKCE
)


@router.get("/auth/login")
async def login(request: Request):
    redirect_uri = request.url_for("auth_callback")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/auth/callback", name="auth_callback")
async def auth_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError:
        # denied consent, CSRF/state mismatch, etc.
        return RedirectResponse(f"{FRONTEND_URL}/?error=login_failed")
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="Could not reach Google")

    userinfo = token.get("userinfo")
    if not userinfo or not userinfo.get("email"):
        return RedirectResponse(f"{FRONTEND_URL}/?error=login_failed")

    request.session["user"] = {
        "email": userinfo["email"],
        "name": userinfo.get("name", ""),
        "exp": time.time() + SESSION_MAX_AGE_SECONDS,
    }
    return RedirectResponse(FRONTEND_URL)


@router.get("/auth/me")
def me(request: Request):
    user = request.session.get("user")
    if not user or user.get("exp", 0) < time.time():
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"email": user["email"], "name": user["name"]}


@router.post("/auth/logout")
def logout(request: Request):
    request.session.clear()
    return {"status": "ok"}


def require_user(request: Request) -> dict:
    """FastAPI dependency: 401s unless the session holds a valid, unexpired user."""
    user = request.session.get("user")
    if not user or user.get("exp", 0) < time.time():
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user
