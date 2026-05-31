"""
ingestion/doc_chunker.py
─────────────────────────
Sliding-window chunker for uploaded documents and on-disk folder ingestion.
400-token window, 50-token overlap, whitespace-split tokenization.
Supports .txt, .md, and .pdf files.
"""

import hashlib
import io
import os
from pathlib import Path
from typing import Generator, TypedDict

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


class ChunkDict(TypedDict):
    chunk_id: str
    doc_id: str
    filename: str
    text: str
    chunk_index: int
    total_chunks: int


def _extract_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n\n".join(pages).strip()
            if len(text) < 50:
                print(f"  ⚠️  PDF '{filename}' yielded very little text — may be scanned/image-based.")
            return text
        except Exception as ex:
            print(f"  ⚠️  PDF extraction failed for '{filename}': {ex}. Falling back to raw decode.")

    return file_bytes.decode("utf-8", errors="replace")


def chunk_text(
    text: str,
    chunk_tokens: int = 400,
    overlap_tokens: int = 50,
) -> list[str]:
    tokens = text.split()
    if not tokens:
        return []

    step = chunk_tokens - overlap_tokens  # 350
    chunks = []
    for i in range(0, len(tokens), step):
        window = tokens[i : i + chunk_tokens]
        if len(window) < 20:
            break
        chunks.append(" ".join(window))

    # If the whole text is shorter than one window, return it as-is
    if not chunks and tokens:
        chunks.append(" ".join(tokens))

    return chunks


def chunk_file(file_bytes: bytes, filename: str) -> list[ChunkDict]:
    text = _extract_text(file_bytes, filename)
    raw_chunks = chunk_text(text)

    if not raw_chunks:
        return []

    doc_id = hashlib.sha256(
        f"{filename}{text[:200]}".encode()
    ).hexdigest()[:12]

    total = len(raw_chunks)
    result: list[ChunkDict] = []
    for i, chunk_text_str in enumerate(raw_chunks):
        result.append(ChunkDict(
            chunk_id=f"{doc_id}_chunk_{i:04d}",
            doc_id=doc_id,
            filename=filename,
            text=chunk_text_str,
            chunk_index=i,
            total_chunks=total,
        ))
    return result


def chunk_files(uploaded_files: list[tuple[str, bytes]]) -> list[ChunkDict]:
    all_chunks: list[ChunkDict] = []
    for filename, file_bytes in uploaded_files:
        chunks = chunk_file(file_bytes, filename)
        all_chunks.extend(chunks)
    return all_chunks


def iter_folder(folder_path: str) -> Generator[tuple[str, int], None, None]:
    """
    Yield (absolute_file_path, total_file_count) for every supported file
    found recursively under folder_path.  total_file_count is emitted once
    at the start as the first tuple with path="".
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder_path}")

    all_files = [
        str(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    yield ("", len(all_files))   # sentinel: total count
    for path in all_files:
        yield (path, len(all_files))


def chunk_file_from_path(file_path: str) -> list[ChunkDict]:
    """Read a file from disk and return its chunks. Memory-safe: one file at a time."""
    path = Path(file_path)
    with open(path, "rb") as f:
        file_bytes = f.read()
    return chunk_file(file_bytes, path.name)
