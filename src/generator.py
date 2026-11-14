"""Answer generation: grounded, cited responses from retrieved context."""
from __future__ import annotations

from typing import Iterator, Sequence

from src.config import Settings, get_settings, require_api_key
from src.retriever import RetrievedChunk

NO_CONTEXT_ANSWER = (
    "I couldn't find anything relevant in the indexed documents to answer this question."
)

SYSTEM_PROMPT = """You are a careful assistant that answers questions about the user's documents.

Rules:
- Use ONLY the numbered context passages provided. Do not use outside knowledge.
- If the context does not contain the answer, say that the documents do not cover it.
- Cite the passages you rely on with their numbers in square brackets, e.g. [1] or [2][3].
- Be concise and factual."""

USER_PROMPT = """Context passages:
{context}

Question: {question}"""


def get_llm(settings: Settings | None = None, model: str | None = None):
    settings = settings or get_settings()
    require_api_key()
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model or settings.chat_model, temperature=0)


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{number}] (source: {chunk.label})\n{chunk.content}"
        for number, chunk in enumerate(chunks, start=1)
    )


def build_messages(question: str, chunks: Sequence[RetrievedChunk]) -> list[tuple[str, str]]:
    return [
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT.format(context=format_context(chunks), question=question.strip())),
    ]


def _text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):  # some models return content blocks
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    return str(content)


def generate_answer(
    question: str,
    chunks: Sequence[RetrievedChunk],
    settings: Settings | None = None,
    llm=None,
) -> str:
    """Generate an answer grounded in `chunks`."""
    if not chunks:
        return NO_CONTEXT_ANSWER
    llm = llm or get_llm(settings)
    return _text(llm.invoke(build_messages(question, chunks))).strip()


def stream_answer(
    question: str,
    chunks: Sequence[RetrievedChunk],
    settings: Settings | None = None,
    llm=None,
) -> Iterator[str]:
    """Yield the answer piece by piece (for live display)."""
    if not chunks:
        yield NO_CONTEXT_ANSWER
        return
    llm = llm or get_llm(settings)
    for piece in llm.stream(build_messages(question, chunks)):
        text = _text(piece)
        if text:
            yield text
