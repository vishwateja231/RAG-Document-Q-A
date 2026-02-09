"""Document ingestion and chunking utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List

from pypdf import PdfReader


def extract_pdf_text(file_path: Path) -> str:
    """Extract text from a PDF file.

    Raises:
        ValueError: if PDF has no extractable text.
    """
    reader = PdfReader(str(file_path))
    pages_text: List[str] = []

    for page in reader.pages:
        pages_text.append(page.extract_text() or "")

    combined = "\n".join(pages_text).strip()
    if not combined:
        raise ValueError("No extractable text found in PDF.")

    return combined


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    """Split text into overlapping chunks for embedding."""
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    cleaned = " ".join(text.split())
    chunks: List[str] = []
    start = 0

    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(cleaned):
            break

        start = end - chunk_overlap

    return chunks
