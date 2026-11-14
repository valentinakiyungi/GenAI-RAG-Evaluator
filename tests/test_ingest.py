import pytest

from src.ingest import (
    find_documents, ingest_directory, ingest_file, load_file, save_upload, split_documents,
)
from src.retriever import list_sources


def make_pdf(path, text):
    """Write a minimal one-page PDF containing `text`."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    path.write_bytes(out)


def test_load_text_and_split(settings, handbook):
    docs = load_file(handbook)
    assert len(docs) == 1 and docs[0].metadata == {"source": "handbook.txt"}
    chunks = split_documents(docs, settings)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= settings.chunk_size for c in chunks)


def test_load_pdf_keeps_page_numbers(tmp_path):
    pdf = tmp_path / "policy.pdf"
    make_pdf(pdf, "Passwords must be at least 14 characters")
    docs = load_file(pdf)
    assert len(docs) == 1
    assert "14 characters" in docs[0].page_content
    assert docs[0].metadata == {"source": "policy.pdf", "page": 1}


def test_unsupported_type_is_rejected(tmp_path):
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"x")
    with pytest.raises(ValueError):
        load_file(path)


def test_ingest_is_idempotent_and_replaces_changed_files(settings, vectorstore, handbook):
    first = ingest_file(handbook, settings, vectorstore)
    assert first.status == "added" and first.chunks > 1
    assert list_sources(settings, vectorstore) == {"handbook.txt": first.chunks}

    again = ingest_file(handbook, settings, vectorstore)
    assert again.status == "skipped"
    assert list_sources(settings, vectorstore) == {"handbook.txt": first.chunks}

    handbook.write_text("Completely new, much shorter content.", encoding="utf-8")
    changed = ingest_file(handbook, settings, vectorstore)
    assert changed.status == "replaced" and changed.chunks == 1
    assert list_sources(settings, vectorstore) == {"handbook.txt": 1}


def test_empty_and_broken_files_are_reported(settings, vectorstore):
    settings.data_dir.mkdir(parents=True)
    (settings.data_dir / "blank.txt").write_text("   \n", encoding="utf-8")
    (settings.data_dir / "broken.pdf").write_bytes(b"this is not a pdf")
    (settings.data_dir / "ignored.docx").write_bytes(b"x")
    reports = {r.source: r for r in ingest_directory(None, settings, vectorstore)}
    assert set(reports) == {"blank.txt", "broken.pdf"}
    assert reports["blank.txt"].status == "empty"
    assert reports["broken.pdf"].status in ("error", "empty")
    assert list_sources(settings, vectorstore) == {}


def test_ingest_directory_creates_missing_folder(settings, vectorstore):
    assert ingest_directory(None, settings, vectorstore) == []
    assert settings.data_dir.is_dir()


def test_save_upload_sanitises_names(settings):
    path = save_upload("../../evil/notes.txt", b"hello", settings)
    assert path == settings.data_dir / "notes.txt"
    assert path.read_bytes() == b"hello"
    assert find_documents(settings.data_dir) == [path]
    with pytest.raises(ValueError):
        save_upload("malware.exe", b"x", settings)
