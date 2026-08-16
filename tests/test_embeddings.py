from unittest.mock import MagicMock, patch

import embeddings


def test_disabled_by_default():
    assert embeddings.is_enabled() is False


def test_rank_by_relevance_passthrough_when_disabled():
    candidates = [b"a", b"b", b"c"]
    result = embeddings.rank_by_relevance("def f(x): return x", candidates)
    assert result == candidates  # identical order, no reordering attempted


def test_cosine_similarity_identical_vectors_is_one():
    assert embeddings._cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert embeddings._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_zero_vector_does_not_divide_by_zero():
    assert embeddings._cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_rank_by_relevance_orders_by_similarity_when_enabled():
    fake_embedder = MagicMock()
    # query embeds to [1, 0]; candidate A close to it, candidate B orthogonal.
    fake_embedder.embed_query.side_effect = lambda text: {
        "query": [1.0, 0.0],
        "close": [0.9, 0.1],
        "far": [0.0, 1.0],
    }[text]

    with patch.object(embeddings, "_embedder", fake_embedder), \
         patch.object(embeddings, "_embedding_cache", {}):
        result = embeddings.rank_by_relevance("query", [b"far", b"close"])

    assert result == [b"close", b"far"]  # more similar one ranked first


def test_rank_by_relevance_returns_original_order_when_query_embedding_fails():
    fake_embedder = MagicMock()
    fake_embedder.embed_query.side_effect = ConnectionError("ollama unreachable")

    with patch.object(embeddings, "_embedder", fake_embedder), \
         patch.object(embeddings, "_embedding_cache", {}):
        candidates = [b"a", b"b"]
        result = embeddings.rank_by_relevance("query", candidates)

    assert result == candidates


def test_rank_by_relevance_deprioritizes_candidate_that_fails_to_embed():
    fake_embedder = MagicMock()

    def _embed_query(text):
        if text == "bad":
            raise ValueError("can't embed this one")
        return {"query": [1.0, 0.0], "good": [1.0, 0.0]}[text]

    fake_embedder.embed_query.side_effect = _embed_query

    with patch.object(embeddings, "_embedder", fake_embedder), \
         patch.object(embeddings, "_embedding_cache", {}):
        result = embeddings.rank_by_relevance("query", [b"bad", b"good"])

    assert result == [b"good", b"bad"]  # the embeddable one ranks above the failure


def test_embed_caches_by_content():
    fake_embedder = MagicMock()
    fake_embedder.embed_query.return_value = [1.0, 2.0]

    with patch.object(embeddings, "_embedder", fake_embedder), \
         patch.object(embeddings, "_embedding_cache", {}):
        embeddings._embed(b"same content")
        embeddings._embed(b"same content")

    fake_embedder.embed_query.assert_called_once()


def test_embed_returns_none_for_empty_content():
    fake_embedder = MagicMock()
    with patch.object(embeddings, "_embedder", fake_embedder), \
         patch.object(embeddings, "_embedding_cache", {}):
        assert embeddings._embed(b"   \n  ") is None
    fake_embedder.embed_query.assert_not_called()
