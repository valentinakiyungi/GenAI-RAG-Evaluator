"""Shared fixtures. The whole suite runs offline: no OpenAI key or network needed."""
from __future__ import annotations

import hashlib
import math
import re

import pytest
from langchain_core.embeddings import Embeddings

from src.config import get_settings
from src.retriever import get_vectorstore

HANDBOOK = """Annual leave
Full-time employees receive 28 days of paid annual leave per calendar year.

Remote work
Employees may work remotely up to three days per week.

Expenses
Business expenses must be submitted within 30 days of purchase with a receipt.
"""


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings: similar wording -> similar vectors."""

    size = 256

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.size
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.size
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Stands in for ChatOpenAI; records the prompts it receives."""

    def __init__(self, reply="Employees get 28 days of leave [1]."):
        self.reply = reply
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return FakeMessage(self.reply)

    def stream(self, messages):
        self.calls.append(messages)
        for word in self.reply.split(" "):
            yield FakeMessage(word + " ")


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("HISTORY_FILE", str(tmp_path / "logs" / "history.jsonl"))
    monkeypatch.setenv("CHUNK_SIZE", "120")
    monkeypatch.setenv("CHUNK_OVERLAP", "20")
    monkeypatch.setenv("TOP_K", "2")
    monkeypatch.setenv("SEARCH_TYPE", "similarity")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return get_settings()


@pytest.fixture
def vectorstore(settings):
    return get_vectorstore(settings, FakeEmbeddings())


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def handbook(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    path = settings.data_dir / "handbook.txt"
    path.write_text(HANDBOOK, encoding="utf-8")
    return path
