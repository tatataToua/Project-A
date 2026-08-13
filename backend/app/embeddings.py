from app.config import EMBEDDING_DIMENSIONS, GEMINI_EMBEDDING_MODEL
from app.llm import client as _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = _client.embeddings.create(
        model=GEMINI_EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    vectors = [item.embedding for item in response.data]
    # A short response would otherwise silently misalign vectors with their
    # texts (zip truncates) or blow up with an opaque unpacking error further up.
    if len(vectors) != len(texts):
        raise ValueError(
            f"Embedding provider returned {len(vectors)} vectors for {len(texts)} inputs"
        )
    return vectors
