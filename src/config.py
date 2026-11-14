"""Central configuration, read from environment variables (and `.env`)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")
SEARCH_TYPES = ("similarity", "mmr")
_PLACEHOLDER_KEYS = {"", "api_key_here", "your_api_key_here"}


class ConfigError(RuntimeError):
    """Raised when the application is not configured correctly."""


def _path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    chat_model: str
    eval_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    search_type: str
    data_dir: Path
    chroma_dir: Path
    collection_name: str
    history_file: Path


def get_settings() -> Settings:
    """Build settings from the current environment."""
    settings = Settings(
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o"),
        eval_model=os.getenv("EVAL_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        chunk_size=_int("CHUNK_SIZE", 1000),
        chunk_overlap=_int("CHUNK_OVERLAP", 200),
        top_k=_int("TOP_K", 4),
        search_type=os.getenv("SEARCH_TYPE", "similarity").strip().lower(),
        data_dir=_path("DATA_DIR", "data"),
        chroma_dir=_path("CHROMA_DIR", "chroma_db"),
        collection_name=os.getenv("COLLECTION_NAME", "rag_documents"),
        history_file=_path("HISTORY_FILE", "logs/history.jsonl"),
    )
    if settings.chunk_size <= 0:
        raise ConfigError("CHUNK_SIZE must be positive")
    if not 0 <= settings.chunk_overlap < settings.chunk_size:
        raise ConfigError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
    if settings.top_k <= 0:
        raise ConfigError("TOP_K must be positive")
    if settings.search_type not in SEARCH_TYPES:
        raise ConfigError(f"SEARCH_TYPE must be one of {SEARCH_TYPES}")
    return settings


def has_api_key() -> bool:
    return os.getenv("OPENAI_API_KEY", "").strip() not in _PLACEHOLDER_KEYS


def require_api_key() -> None:
    if not has_api_key():
        raise ConfigError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
