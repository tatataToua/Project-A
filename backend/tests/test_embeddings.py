from types import SimpleNamespace

import pytest

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


class _ShortEmbeddingsAPI:
    """Returns fewer vectors than inputs -- e.g. a truncated provider response."""

    def create(self, **kwargs):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


def test_embed_texts_raises_when_provider_returns_too_few_vectors(monkeypatch):
    monkeypatch.setattr(embeddings, "_client", SimpleNamespace(embeddings=_ShortEmbeddingsAPI()))

    with pytest.raises(ValueError, match="1 vectors for 2 inputs"):
        embeddings.embed_texts(["hello", "world"])
