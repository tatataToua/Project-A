"""First real-model test downloads cross-encoder/ms-marco-MiniLM-L-6-v2 on first
run (network + local Hugging Face cache) -- subsequent runs are fast."""
from app import reranking


def test_rerank_chunks_returns_empty_list_for_no_candidates():
    assert reranking.rerank_chunks("query", [], k=5) == []


def test_rerank_chunks_orders_by_score_descending(monkeypatch):
    class _FakeModel:
        def predict(self, pairs):
            return [1.0 if "match" in chunk_text else 0.0 for _q, chunk_text in pairs]

    monkeypatch.setattr(reranking, "_get_model", lambda: _FakeModel())

    candidates = [
        (1, "a.md", "irrelevant text"),
        (2, "b.md", "this is a match"),
        (3, "c.md", "also irrelevant"),
    ]
    result = reranking.rerank_chunks("query", candidates, k=2)

    assert result == [("b.md", "this is a match"), ("a.md", "irrelevant text")]


def test_rerank_chunks_truncates_to_k(monkeypatch):
    class _FakeModel:
        def predict(self, pairs):
            return [float(i) for i in range(len(pairs))]

    monkeypatch.setattr(reranking, "_get_model", lambda: _FakeModel())

    candidates = [(1, "a.md", "a"), (2, "b.md", "b"), (3, "c.md", "c")]
    result = reranking.rerank_chunks("query", candidates, k=1)

    assert result == [("c.md", "c")]


def test_rerank_chunks_uses_the_real_cross_encoder_model():
    candidates = [
        (1, "about.md", "The restaurant is open from 5pm to 10pm on weekdays."),
        (2, "about.md", "Our chef trained in Paris for six years."),
    ]
    result = reranking.rerank_chunks("What time do you open?", candidates, k=1)

    assert result == [("about.md", "The restaurant is open from 5pm to 10pm on weekdays.")]
