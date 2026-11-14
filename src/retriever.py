"""Vector store access and semantic retrieval (ChromaDB + OpenAI embeddings)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from src.config import SEARCH_TYPES, Settings, get_settings, require_api_key


@dataclass
class RetrievedChunk:
    content: str
    source: str
    page: int | None = None
    score: float | None = None  # relevance in [0, 1]; None for MMR results

    @property
    def label(self) -> str:
        return f"{self.source}, p. {self.page}" if self.page else self.source

    def to_dict(self) -> dict:
        return asdict(self)


def get_embeddings(settings: Settings | None = None):
    settings = settings or get_settings()
    require_api_key()
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=settings.embedding_model)


def get_vectorstore(settings: Settings | None = None, embeddings=None):
    """Open (or create) the persistent Chroma collection."""
    settings = settings or get_settings()
    from langchain_chroma import Chroma

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings or get_embeddings(settings),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def retrieve(
    query: str,
    k: int | None = None,
    search_type: str | None = None,
    settings: Settings | None = None,
    vectorstore=None,
) -> list[RetrievedChunk]:
    """Return the chunks most relevant to `query`."""
    settings = settings or get_settings()
    k = k or settings.top_k
    search_type = search_type or settings.search_type
    if search_type not in SEARCH_TYPES:
        raise ValueError(f"search_type must be one of {SEARCH_TYPES}")
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    vectorstore = vectorstore or get_vectorstore(settings)

    if search_type == "mmr":
        docs = vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=max(k * 4, 20))
        pairs = [(doc, None) for doc in docs]
    else:
        pairs = vectorstore.similarity_search_with_relevance_scores(query, k=k)

    chunks = []
    for doc, score in pairs:
        meta = doc.metadata or {}
        chunks.append(
            RetrievedChunk(
                content=doc.page_content,
                source=str(meta.get("source", "unknown")),
                page=meta.get("page"),
                score=None if score is None else round(max(0.0, min(1.0, float(score))), 4),
            )
        )
    return chunks


def list_sources(settings: Settings | None = None, vectorstore=None) -> dict[str, int]:
    """Map each indexed document name to its number of chunks."""
    vectorstore = vectorstore or get_vectorstore(settings)
    counts: dict[str, int] = {}
    for meta in vectorstore.get(include=["metadatas"]).get("metadatas") or []:
        name = str((meta or {}).get("source", "unknown"))
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def delete_source(source: str, settings: Settings | None = None, vectorstore=None) -> int:
    """Remove every chunk of one document. Returns the number of chunks removed."""
    vectorstore = vectorstore or get_vectorstore(settings)
    ids = vectorstore.get(where={"source": source}, include=["metadatas"]).get("ids") or []
    if ids:
        vectorstore.delete(ids=list(ids))
    return len(ids)


def reset_vectorstore(settings: Settings | None = None, vectorstore=None) -> None:
    """Delete the whole collection."""
    vectorstore = vectorstore or get_vectorstore(settings)
    vectorstore.delete_collection()
