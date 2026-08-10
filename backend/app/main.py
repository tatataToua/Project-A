import logging
import time

from fastapi import Depends, FastAPI, HTTPException
from openai import OpenAIError
from pydantic import BaseModel
from sqlalchemy import select
from starlette.middleware.sessions import SessionMiddleware

from app import auth, ratelimit, workflow
from app.config import (
    SESSION_COOKIE_SECURE,
    SESSION_MAX_AGE_SECONDS,
    SESSION_SECRET,
)
from app.db import SessionLocal
from app.models import EmbeddingChunk, Tenant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("askme")

app = FastAPI(title="Ask Me")

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    max_age=SESSION_MAX_AGE_SECONDS,
    same_site="lax",
    https_only=SESSION_COOKIE_SECURE,
)

app.include_router(auth.router)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat/{tenant_slug}", response_model=ChatResponse)
def chat(
    tenant_slug: str,
    req: ChatRequest,
    user: dict = Depends(auth.require_user),
    _: None = Depends(ratelimit.enforce_chat_rate_limit),
) -> ChatResponse:
    session = SessionLocal()
    try:
        tenant = session.scalar(select(Tenant).where(Tenant.slug == tenant_slug))
        if tenant is None:
            raise HTTPException(status_code=404, detail="Unknown tenant.")

        has_content = session.scalar(
            select(EmbeddingChunk.id).where(EmbeddingChunk.tenant_id == tenant.id).limit(1)
        )
        if has_content is None:
            return ChatResponse(reply="I don't have any information loaded yet -- check back soon.")
        tenant_id = tenant.id
    finally:
        session.close()

    start = time.monotonic()
    try:
        answer = workflow.run_chat_workflow(tenant_id, req.message)
    except OpenAIError:
        logger.exception("Gemini request failed for user=%s", user["email"])
        raise HTTPException(
            status_code=502,
            detail="Could not reach the assistant right now — try again shortly.",
        )

    logger.info(
        "chat request user=%s tenant=%s elapsed=%.2fs", user["email"], tenant_slug, time.monotonic() - start
    )
    return ChatResponse(reply=answer)
