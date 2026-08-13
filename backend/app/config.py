import logging
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger("askme")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.environ.get(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "768"))

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg://askme:askme@localhost:5432/askme"
)

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")
SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true"
SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 7  # 7 days
_MIN_SESSION_SECRET_LENGTH = 32


def _resolve_session_secret() -> str:
    """The key that signs the session cookie -- i.e. the whole of auth.

    An empty or trivially short key means anyone can mint a cookie claiming any
    email, so it is never allowed to fall back to a constant. When it's missing
    we refuse to start with `SESSION_COOKIE_SECURE=true` (the production
    signal), and otherwise use a random per-process key so local dev still runs
    -- at the cost of every session dying on restart.
    """
    secret = os.environ.get("SESSION_SECRET", "")
    if len(secret) >= _MIN_SESSION_SECRET_LENGTH:
        return secret

    problem = "is not set" if not secret else (
        f"is shorter than {_MIN_SESSION_SECRET_LENGTH} characters"
    )
    if SESSION_COOKIE_SECURE:
        raise RuntimeError(
            f"SESSION_SECRET {problem}. Generate one with "
            '`python -c "import secrets; print(secrets.token_hex(32))"` and set it in '
            "backend/.env -- session cookies cannot be signed securely without it."
        )
    logger.warning(
        "SESSION_SECRET %s; using a random per-process key. Logins will not survive a "
        "restart, and this is refused outright when SESSION_COOKIE_SECURE=true.",
        problem,
    )
    return secrets.token_hex(32)


SESSION_SECRET = _resolve_session_secret()

CHAT_MESSAGE_MAX_LENGTH = int(os.environ.get("CHAT_MESSAGE_MAX_LENGTH", "2000"))

RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))

RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "5"))
RETRIEVAL_OVERFETCH_K = int(os.environ.get("RETRIEVAL_OVERFETCH_K", "15"))
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONTENT_DIR = REPO_ROOT / "docs" / "content"
INSTRUCTIONS_FILE = Path(
    os.environ.get("INSTRUCTIONS_FILE", str(REPO_ROOT / "backend" / "instructions.txt"))
)
