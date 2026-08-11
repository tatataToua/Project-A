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
    RATE_LIMIT_MAX_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
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


@app.on_event("startup")
def log_startup_config() -> None:
    about_path = CONTENT_DIR / "about.md"
    logger.info(
        "startup config: model=%s base_url=%s api_key_set=%s content_dir=%s "
        "about_md_found=%s rate_limit=%d/%ds",
        GEMINI_MODEL,
        GEMINI_BASE_URL,
        bool(GEMINI_API_KEY),
        CONTENT_DIR,
        about_path.exists(),
        RATE_LIMIT_MAX_REQUESTS,
        RATE_LIMIT_WINDOW_SECONDS,
    )


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


def build_system_prompt() -> str:
    about_path = CONTENT_DIR / "about.md"
    about = about_path.read_text(encoding="utf-8") if about_path.exists() else ""
    return (
        "You are an AI assistant that answers questions using only the background "
        "information provided below. Answer in a natural voice appropriate to what's "
        "described (e.g. first person for a person's bio, \"we\" for a business). Stay "
        "strictly grounded in the provided background, and clearly say when a question "
        "isn't covered by it rather than guessing.\n\n"
        f"--- BACKGROUND ---\n{about}"
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
    logger.info(
        "chat request user=%s model=%s message_len=%d",
        user["email"],
        GEMINI_MODEL,
        len(req.message),
    )
    try:
        completion = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": req.message},
            ],
        )
    except OpenAIError as e:
        logger.exception(
            "Gemini request failed user=%s model=%s elapsed=%.2fs error_type=%s",
            user["email"],
            GEMINI_MODEL,
            time.monotonic() - start,
            type(e).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="Could not reach the assistant right now — try again shortly.",
        )

    usage = completion.usage
    logger.info(
        "chat response user=%s model=%s elapsed=%.2fs prompt_tokens=%s "
        "completion_tokens=%s total_tokens=%s",
        user["email"],
        GEMINI_MODEL,
        time.monotonic() - start,
        usage.prompt_tokens if usage else "?",
        usage.completion_tokens if usage else "?",
        usage.total_tokens if usage else "?",
    )
    return ChatResponse(reply=completion.choices[0].message.content or "")
