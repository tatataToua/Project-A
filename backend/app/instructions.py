# backend/app/instructions.py
import logging

from app.config import INSTRUCTIONS_FILE

logger = logging.getLogger("askme")


def get_custom_instructions() -> str:
    """Operator instructions if a readable file exists, else "".

    An unreadable or non-UTF-8 file is a local operator mistake, not a reason to
    fail the chat request -- it's logged and treated as "no extra instructions".
    """
    try:
        content = INSTRUCTIONS_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except (OSError, UnicodeDecodeError):
        logger.warning("Could not read %s; ignoring custom instructions", INSTRUCTIONS_FILE, exc_info=True)
        return ""
    return content.strip()
