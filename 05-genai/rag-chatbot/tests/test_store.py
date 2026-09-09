import numpy as np
import pytest

from rag.embeddings import EmbeddingCache, cache_key, embed_chunks, embed_query, normalise
from rag.store import VectorStore, tokenise

from conftest import DIMENSIONS


# --------------------------------------------------------------- embeddings


def test_embed_chunks_returns_one_vector_each(chunks, client):
    vectors = embed_chunks(chunks, client, dimensions=DIMENSIONS)
    assert vectors.shape == (len(chunks), DIMENSIONS)


def test_documents_are_embedded_as_documents(chunks, client, transport):
    embed_chunks(chunks, client, dimensions=DIMENSIONS)
    assert transport.embed_calls[0]["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"


def test_queries_are_embedded_as_queries(client, transport):
    embed_query("what was the cutoff?", client, dimensions=DIMENSIONS)
    assert transport.embed_calls[0]["requests"][0]["taskType"] == "RETRIEVAL_QUERY"


def test_heading_context_is_embedded(chunks, client, transport):
    """Chunks embed with their heading path, not the bare text."""
    embed_chunks(chunks, client, dimensions=DIMENSIONS)
    sent = [r["content"]["parts"][0]["text"] for r in transport.embed_calls[0]["requests"]]
    assert any(">" in text for text in sent)


def test_empty_chunk_list(client):
    assert embed_chunks([], client, dimensions=DIMENSIONS).shape == (0, DIMENSIONS)


def test_batching_splits_large_inputs(chunks, client, transport):
    embed_chunks(chunks, client, dimensions=DIMENSIONS, batch_size=2)
    assert len(transport.embed_calls) >= len(chunks) / 2


# --------------------------------------------------------------------- cache


def test_cache_key_changes_with_text():
    assert cache_key("a", "m", 8) != cache_key("b", "m", 8)


def test_cache_key_changes_with_model():
    """Switching embedding models must invalidate everything automatically."""
    assert cache_key("a", "model-1", 8) != cache_key("a", "model-2", 8)


def test_cache_avoids_a_second_api_call(chunks, client, transport, tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.json")
    embed_chunks(chunks, client, cache=cache, dimensions=DIMENSIONS)
    first = len(transport.embed_calls)

    embed_chunks(chunks, client, cache=cache, dimensions=DIMENSIONS)
    assert len(transport.embed_calls) == first
    assert cache.hits >= len(chunks)


def test_cache_survives_a_reload(chunks, client, tmp_path):
    path = tmp_path / "cache.json"
    embed_chunks(chunks, client, cache=EmbeddingCache(path), dimensions=DIMENSIONS)
    assert len(EmbeddingCache(path)) == len(chunks)


def test_corrupt_cache_is_treated_as_cold(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{not json")
    assert len(EmbeddingCache(path)) == 0


def test_normalise_gives_unit_rows():
    matrix = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    assert np.allclose(np.linalg.norm(normalise(matrix), axis=1), 1.0)


def test_normalise_survives_a_zero_vector():
    assert np.isfinite(normalise(np.zeros((1, 4), dtype=np.float32))).all()


# ------------------------------------------------------------- dense search


def test_store_length(store, chunks):
    assert len(store) == len(chunks)


def test_mismatched_lengths_are_rejected(chunks):
    with pytest.raises(ValueError, match="chunks but"):
        VectorStore(chunks, np.zeros((len(chunks) + 1, DIMENSIONS), dtype=np.float32))


def test_search_returns_k_hits(store, client):
    hits = store.search(embed_query("campaign cost", client, dimensions=DIMENSIONS), k=3)
    assert len(hits) == 3


def test_search_is_ranked(store, client):
    hits = store.search(embed_query("fraud threshold", client, dimensions=DIMENSIONS), k=5)
    assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]


def test_dense_search_finds_the_right_topic(store, client):
    hits = store.search(embed_query("retention campaign contact", client, dimensions=DIMENSIONS), k=3)
    assert any(h.chunk.doc_id == "churn" for h in hits)


def test_k_larger_than_the_corpus_is_safe(store, client):
    hits = store.search(embed_query("anything", client, dimensions=DIMENSIONS), k=999)
    assert len(hits) <= len(store)


# ----------------------------------------------------------- keyword search


def test_tokenise_drops_stopwords():
    assert "the" not in tokenise("the fraud threshold")
    assert "fraud" in tokenise("the fraud threshold")


def test_keyword_search_finds_an_exact_number(store):
    """Where dense retrieval is weak: a literal rare token."""
    hits = store.keyword_search("0.1141", k=3)
    assert any("0.1141" in h.text for h in hits)


def test_keyword_search_on_stopwords_only_returns_nothing(store):
    assert store.keyword_search("the and of", k=3) == []


def test_keyword_search_is_ranked(store):
    hits = store.keyword_search("precision recall threshold", k=4)
    assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)


# ------------------------------------------------------------ hybrid search


def test_hybrid_returns_k(store, client):
    hits = store.hybrid_search(
        "fraud threshold", embed_query("fraud threshold", client, dimensions=DIMENSIONS), k=3
    )
    assert len(hits) == 3


def test_hybrid_records_both_source_ranks(store, client):
    hits = store.hybrid_search(
        "precision recall", embed_query("precision recall", client, dimensions=DIMENSIONS), k=5
    )
    assert any(h.dense_rank is not None for h in hits)
    assert any(h.keyword_rank is not None for h in hits)


def test_hybrid_recovers_a_literal_dense_search_can_miss(store, client):
    """The reason hybrid exists: exact tokens that embeddings smear over."""
    query = "0.1141"
    hits = store.hybrid_search(query, embed_query(query, client, dimensions=DIMENSIONS), k=5)
    assert any("0.1141" in h.text for h in hits)


def test_hybrid_deduplicates(store, client):
    hits = store.hybrid_search(
        "fraud", embed_query("fraud", client, dimensions=DIMENSIONS), k=8
    )
    assert len({h.chunk.chunk_id for h in hits}) == len(hits)


def test_hybrid_ranks_are_sequential(store, client):
    hits = store.hybrid_search(
        "churn", embed_query("churn", client, dimensions=DIMENSIONS), k=4
    )
    assert [h.rank for h in hits] == [1, 2, 3, 4]


# ------------------------------------------------------------- persistence


def test_save_and_load_round_trip(store, tmp_path):
    store.save(tmp_path / "index")
    loaded = VectorStore.load(tmp_path / "index")
    assert len(loaded) == len(store)
    assert loaded.chunks[0].chunk_id == store.chunks[0].chunk_id
    assert np.allclose(loaded.vectors, store.vectors)


def test_loaded_store_still_searches(store, client, tmp_path):
    store.save(tmp_path / "index")
    loaded = VectorStore.load(tmp_path / "index")
    hits = loaded.search(embed_query("fraud", client, dimensions=DIMENSIONS), k=2)
    assert len(hits) == 2


def test_loading_a_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no index"):
        VectorStore.load(tmp_path / "nothing")
