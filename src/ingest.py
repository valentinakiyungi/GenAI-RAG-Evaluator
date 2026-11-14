"""Document ingestion: load PDF/text files, chunk them and store embeddings.

Run as a script to index everything in the data directory:

    python src/ingest.py            # index new / changed files in data/
    python src/ingest.py --reset    # wipe the vector store first
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python src/ingest.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import hashlib
from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import SUPPORTED_EXTENSIONS, ConfigError, Settings, get_settings
from src.retriever import delete_source, get_vectorstore, reset_vectorstore

_BATCH_SIZE = 200


@dataclass
class IngestReport:
    source: str
    status: str  # added | replaced | skipped | empty | error
    chunks: int = 0
    detail: str = ""


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def load_file(path: Path) -> list[Document]:
    """Read one file into page-level documents (text files are a single page)."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type {suffix!r}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(
                    Document(page_content=text, metadata={"source": path.name, "page": number})
                )
        return pages
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [Document(page_content=text, metadata={"source": path.name})] if text else []


def split_documents(docs: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    return [chunk for chunk in splitter.split_documents(docs) if chunk.page_content.strip()]


def ingest_file(path: Path, settings: Settings | None = None, vectorstore=None) -> IngestReport:
    """Index one file. Unchanged files are skipped; changed files are replaced."""
    settings = settings or get_settings()
    path = Path(path)
    name = path.name
    try:
        vectorstore = vectorstore or get_vectorstore(settings)
        digest = file_hash(path)
        existing = vectorstore.get(where={"source": name}, include=["metadatas"])
        old_hashes = {meta.get("doc_hash") for meta in existing.get("metadatas") or []}
        if existing.get("ids") and old_hashes == {digest}:
            return IngestReport(name, "skipped", len(existing["ids"]), "already indexed")

        chunks = split_documents(load_file(path), settings)
        if not chunks:
            return IngestReport(
                name, "empty", 0, "no extractable text (scanned PDFs need OCR first)"
            )
        for index, chunk in enumerate(chunks):
            chunk.metadata.update({"doc_hash": digest, "chunk": index})
        ids = [f"{digest}-{index}" for index in range(len(chunks))]

        replaced = bool(existing.get("ids"))
        if replaced:
            delete_source(name, settings, vectorstore)
        for start in range(0, len(chunks), _BATCH_SIZE):
            vectorstore.add_documents(
                chunks[start : start + _BATCH_SIZE], ids=ids[start : start + _BATCH_SIZE]
            )
        return IngestReport(name, "replaced" if replaced else "added", len(chunks))
    except ConfigError:
        raise
    except Exception as exc:  # report per-file failures without stopping a batch
        return IngestReport(name, "error", 0, f"{type(exc).__name__}: {exc}")


def find_documents(directory: Path) -> list[Path]:
    return sorted(
        p for p in Path(directory).rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def ingest_paths(
    paths: Iterable[Path], settings: Settings | None = None, vectorstore=None
) -> list[IngestReport]:
    settings = settings or get_settings()
    paths = list(paths)
    if not paths:
        return []
    vectorstore = vectorstore or get_vectorstore(settings)
    return [ingest_file(path, settings, vectorstore) for path in paths]


def ingest_directory(
    directory: Path | None = None, settings: Settings | None = None, vectorstore=None
) -> list[IngestReport]:
    settings = settings or get_settings()
    directory = Path(directory or settings.data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return ingest_paths(find_documents(directory), settings, vectorstore)


def save_upload(filename: str, content: bytes, settings: Settings | None = None) -> Path:
    """Store an uploaded file in the data directory and return its path."""
    settings = settings or get_settings()
    name = Path(filename.replace("\\", "/")).name
    if not name or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file {filename!r}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    target = settings.data_dir / name
    target.write_bytes(content)
    return target


def build_vector_db(reset: bool = False, data_dir: Path | None = None) -> list[IngestReport]:
    """Index the data directory and print a summary (CLI entry point)."""
    settings = get_settings()
    directory = Path(data_dir or settings.data_dir)
    if reset:
        reset_vectorstore(settings)
        print("Vector store cleared.")
    reports = ingest_directory(directory, settings)
    if not reports:
        print(f"No documents found in '{directory}'. Add PDF, TXT or MD files and re-run.")
        return reports
    for report in reports:
        note = f" ({report.detail})" if report.detail else ""
        print(f"  [{report.status:>8}] {report.source}: {report.chunks} chunks{note}")
    indexed = sum(r.chunks for r in reports if r.status in ("added", "replaced"))
    print(f"Embedded {indexed} new chunks from {len(reports)} file(s) into ChromaDB.")
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Index documents into the vector store.")
    parser.add_argument("--data-dir", type=Path, help="folder to index (default: DATA_DIR)")
    parser.add_argument("--reset", action="store_true", help="clear the vector store first")
    args = parser.parse_args()
    try:
        reports = build_vector_db(reset=args.reset, data_dir=args.data_dir)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    return 1 if any(r.status == "error" for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
