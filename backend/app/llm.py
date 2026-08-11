"""Single shared OpenAI-compatible client.

Everything that talks to an LLM (`embeddings.py`, `workflow.py`) imports this one
client, so pointing the app at a different provider is a change here plus the
matching `config.py` values -- not a hunt for scattered constructions.
"""
from openai import OpenAI

from app.config import GEMINI_API_KEY, GEMINI_BASE_URL

client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)
