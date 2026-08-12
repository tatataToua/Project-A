# backend/app/instructions.py
from app.config import INSTRUCTIONS_FILE


def get_custom_instructions() -> str:
    try:
        content = INSTRUCTIONS_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    return content.strip()
