import logging
import time

from fastapi import Depends, FastAPI, HTTPException
from openai import OpenAI, OpenAIError
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from app import auth, ratelimit
from app.config import (
    CONTENT_DIR,
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    GEMINI_MODEL,
    SESSION_COOKIE_SECURE,
    SESSION_MAX_AGE_SECONDS,
    SESSION_SECRET,
)

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

# Gemini's free tier (no credit card, 1500 req/day) exposes an OpenAI-compatible
# endpoint, so the OpenAI SDK works unchanged — only base_url/model/key differ.
# Swapping to real OpenAI later is a one-line change back to OpenAI(api_key=...).
client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


def build_system_prompt() -> str:
    bio_path = CONTENT_DIR / "bio.md"
    bio = bio_path.read_text(encoding="utf-8") if bio_path.exists() else ""
    return (
        "You are an AI assistant speaking on behalf of the person described below. "
        "Answer questions about their background, skills, and experience in first person, "
        "as if you were their professional voice. Stay grounded in the provided background "
        "and say when something isn't covered by it.\n\n"
        f"--- BACKGROUND ---\n{bio}"
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    user: dict = Depends(auth.require_user),
    _: None = Depends(ratelimit.enforce_chat_rate_limit),
) -> ChatResponse:
    start = time.monotonic()
    try:
        completion = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": req.message},
            ],
        )
    except OpenAIError:
        logger.exception("Gemini request failed for user=%s", user["email"])
        raise HTTPException(
            status_code=502,
            detail="Could not reach the assistant right now — try again shortly.",
        )

    logger.info(
        "chat request user=%s elapsed=%.2fs", user["email"], time.monotonic() - start
    )
    return ChatResponse(reply=completion.choices[0].message.content or "")
