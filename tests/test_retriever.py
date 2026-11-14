import pytest

from src.ingest import ingest_file
from src.retriever import delete_source, list_sources, reset_vectorstore, retrieve


@pytest.fixture
def indexed(settings, vectorstore, handbook):
    ingest_file(handbook, settings, vectorstore)
    return vectorstore


def test_most_relevant_chunk_comes_first(settings, indexed):
    chunks = retrieve("How many days of paid annual leave?", settings=settings, vectorstore=indexed)
    assert len(chunks) == settings.top_k
    assert "28 days" in chunks[0].content
    assert chunks[0].source == "handbook.txt"
    assert all(0.0 <= c.score <= 1.0 for c in chunks)
    assert chunks[0].score >= chunks[1].score


def test_k_and_mmr(settings, indexed):
    assert len(retrieve("remote work", k=1, settings=settings, vectorstore=indexed)) == 1
    chunks = retrieve("remote work", k=2, search_type="mmr", settings=settings, vectorstore=indexed)
    assert len(chunks) == 2 and all(c.score is None for c in chunks)


def test_invalid_input(settings, indexed):
    with pytest.raises(ValueError):
        retrieve("   ", settings=settings, vectorstore=indexed)
    with pytest.raises(ValueError):
        retrieve("x", search_type="magic", settings=settings, vectorstore=indexed)


def test_empty_store_returns_nothing(settings, vectorstore):
    assert retrieve("anything", settings=settings, vectorstore=vectorstore) == []


def test_delete_and_reset(settings, indexed):
    assert delete_source("missing.txt", settings, indexed) == 0
    assert delete_source("handbook.txt", settings, indexed) > 0
    assert list_sources(settings, indexed) == {}
    reset_vectorstore(settings, indexed)


def test_chunk_label():
    from src.retriever import RetrievedChunk

    assert RetrievedChunk("x", "a.pdf", page=3).label == "a.pdf, p. 3"
    assert RetrievedChunk("x", "a.txt").label == "a.txt"
