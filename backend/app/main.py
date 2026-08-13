import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from openai import OpenAIError
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.sessions import SessionMiddleware

from app import auth, ratelimit, tracing
from app.config import (
    SESSION_COOKIE_SECURE,
    SESSION_MAX_AGE_SECONDS,
    SESSION_SECRET,
)
from app.db import SessionLocal
from app.models import EmbeddingChunk, Tenant, create_tables

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
    except SQLAlchemyError:
        # Postgres unreachable or the schema is missing -- a dependency outage,
        # not an unexplained 500 with a stack trace in the response.
        logger.exception("Tenant lookup failed for slug=%s", tenant_slug)
        raise HTTPException(
            status_code=503,
            detail="The assistant is temporarily unavailable — try again shortly.",
        )
    finally:
        session.close()

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
    except Exception:
        # Anything else is a bug rather than an outage: log it with the full
        # traceback (so it is never silently lost) and return a generic 500
        # instead of leaking internals to the browser.
        logger.exception("Unexpected chat failure for user=%s", user["email"])
        raise HTTPException(status_code=500, detail="Something went wrong handling that message.")

    return ChatResponse(reply=answer)
