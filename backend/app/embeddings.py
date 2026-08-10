from openai import OpenAI

from app.config import EMBEDDING_DIMENSIONS, GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_EMBEDDING_MODEL

_client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = _client.embeddings.create(
        model=GEMINI_EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return [item.embedding for item in response.data]
