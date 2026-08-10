import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg://askme:askme@localhost:5432/askme"
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONTENT_DIR = REPO_ROOT / "docs" / "content"
