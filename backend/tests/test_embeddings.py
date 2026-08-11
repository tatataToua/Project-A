from types import SimpleNamespace

from app import embeddings


class _FakeEmbeddingsAPI:
    def __init__(self):
        self.last_call = None

    def create(self, **kwargs):
        self.last_call = kwargs
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3]) for _ in kwargs["input"]]
        )


def test_embed_texts_returns_one_vector_per_input(monkeypatch):
    fake_client = SimpleNamespace(embeddings=_FakeEmbeddingsAPI())
    monkeypatch.setattr(embeddings, "_client", fake_client)

    result = embeddings.embed_texts(["hello", "world"])

    assert result == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


def test_embed_texts_calls_configured_model_and_dimensions(monkeypatch):
    fake_client = SimpleNamespace(embeddings=_FakeEmbeddingsAPI())
    monkeypatch.setattr(embeddings, "_client", fake_client)
    monkeypatch.setattr(embeddings, "GEMINI_EMBEDDING_MODEL", "test-embed-model")
    monkeypatch.setattr(embeddings, "EMBEDDING_DIMENSIONS", 3)

    embeddings.embed_texts(["hello"])

    assert fake_client.embeddings.last_call == {
        "model": "test-embed-model",
        "input": ["hello"],
        "dimensions": 3,
    }
