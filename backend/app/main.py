import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path
from openai import OpenAIError
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.sessions import SessionMiddleware

from app import auth, ratelimit, tracing
from app.config import (
    CHAT_MESSAGE_MAX_LENGTH,
    SESSION_COOKIE_SECURE,
    SESSION_MAX_AGE_SECONDS,
    SESSION_SECRET,
)
from app.db import session_scope
from app.models import EmbeddingChunk, create_tables
from app.tenants import get_tenant_by_slug

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("askme")

tracing.instrument_client()


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    yield


app = FastAPI(title="Ask Me", lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    max_age=SESSION_MAX_AGE_SECONDS,
    same_site="lax",
    https_only=SESSION_COOKIE_SECURE,
)

app.include_router(auth.router)


TENANT_SLUG_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"


class ChatRequest(BaseModel):
    # Bounded because the message goes straight into LLM prompts: an unbounded
    # one is a per-request token-cost amplifier for any logged-in user.
    message: str = Field(min_length=1, max_length=CHAT_MESSAGE_MAX_LENGTH)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        return stripped


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat/{tenant_slug}", response_model=ChatResponse)
def chat(
    tenant_slug: Annotated[str, Path(pattern=TENANT_SLUG_PATTERN, max_length=64)],
    req: ChatRequest,
    user: dict = Depends(auth.require_user),
    _: None = Depends(ratelimit.enforce_chat_rate_limit),
) -> ChatResponse:
    with session_scope() as session:
        tenant = get_tenant_by_slug(session, tenant_slug)
        if tenant is None:
            raise HTTPException(status_code=404, detail="Unknown tenant.")

        has_content = session.scalar(
            select(EmbeddingChunk.id).where(EmbeddingChunk.tenant_id == tenant.id).limit(1)
        )
        if has_content is None:
            return ChatResponse(reply="I don't have any information loaded yet -- check back soon.")
        tenant_id = tenant.id

    try:
        answer, _ = tracing.trace_turn(tenant_id, req.message)
    except (OpenAIError, SQLAlchemyError):
        # The workflow both calls the LLM and hits the database (retrieval) --
        # either failing is an assistant failure, not a bug to leak as a raw 500.
        logger.exception("Chat workflow failed for user=%s", user["email"])
        raise HTTPException(
            status_code=502,
            detail="Could not reach the assistant right now — try again shortly.",
        )

    return ChatResponse(reply=answer)
