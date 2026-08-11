from app.config import EMBEDDING_DIMENSIONS, GEMINI_EMBEDDING_MODEL
from app.llm import client as _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = _client.embeddings.create(
        model=GEMINI_EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return [item.embedding for item in response.data]
