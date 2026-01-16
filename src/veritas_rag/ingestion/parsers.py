"""
Document parsers for Veritas RAG.

MVP: PDF and TXT parsers only.
"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from veritas_rag.core.contracts import Document, Page
from veritas_rag.core.ids import generate_doc_id, generate_doc_uid
from veritas_rag.ingestion.normalizer import normalize_text


def parse_document(file_path: Path, corpus_root: Path) -> Optional[Document]:
    """
    Parse a document based on file extension.

    Args:
        file_path: Path to the document file
        corpus_root: Root directory of the corpus (for relative paths)

    Returns:
        Document object or None if parsing fails
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return parse_pdf(file_path, corpus_root)
    elif suffix in [".txt", ".text"]:
        return parse_text(file_path, corpus_root)
    else:
        # MVP: only PDF and TXT supported
        return None


def parse_pdf(file_path: Path, corpus_root: Path) -> Optional[Document]:
    """
    Parse a PDF document using PyMuPDF.

    Args:
        file_path: Path to PDF file
        corpus_root: Root directory of corpus

    Returns:
        Document object or None if parsing fails
    """
    if fitz is None:
        raise ImportError("PyMuPDF (pymupdf) is required for PDF parsing")

    try:
        doc = fitz.open(file_path)
        pages = []
        raw_text_parts = []
        current_offset = 0

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            raw_text_parts.append(text)

            # Calculate page offsets in normalized text
            # We'll normalize after collecting all text
            page_start = current_offset
            # Approximate end (will be adjusted after normalization)
            page_end = current_offset + len(text)
            current_offset = page_end

            pages.append(
                Page(
                    page_number=page_num,
                    text=text,
                    offset_start=page_start,  # Will be adjusted
                    offset_end=page_end,  # Will be adjusted
                )
            )

        # Combine all text
        raw_text = "\n".join(raw_text_parts)

        # Normalize text (preserves newlines)
        normalized_text = normalize_text(raw_text)

        # Recalculate page offsets in normalized text
        # Build normalized_text incrementally to match offset math exactly
        normalized_text_parts = []
        offset = 0

        for idx, page in enumerate(pages):
            normalized_page_text = normalize_text(page.text)

            # If this is not the first page, add separator BEFORE this page
            if idx > 0:
                normalized_text_parts.append("\n")  # Literally append the separator
                offset += 1  # Increment offset to match

            # Set page offset start
            page.offset_start = offset

            # Append page text and update offset
            normalized_text_parts.append(normalized_page_text)
            offset += len(normalized_page_text)

            # Set page offset end
            page.offset_end = offset

        # Join to create final normalized_text (this should match the offset math exactly)
        normalized_text = "".join(normalized_text_parts)

        # Get relative path
        rel_path = str(file_path.relative_to(corpus_root))

        # Generate document UID (stable) and ID (versioned)
        doc_uid = generate_doc_uid(rel_path)
        normalized_text_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        doc_id = generate_doc_id(doc_uid, normalized_text_hash)

        # Extract title (first line or filename)
        title = doc.metadata.get("title") or file_path.stem

        return Document(
            doc_uid=doc_uid,
            doc_id=doc_id,
            source_path=rel_path,
            raw_text=raw_text,
            normalized_text=normalized_text,
            title=title,
            pages=pages,
            extracted_at=datetime.now(),
            metadata={"pages": len(pages)},
        )

    except Exception as e:
        # Log error and return None
        print(f"Error parsing PDF {file_path}: {e}")
        return None
    finally:
        if "doc" in locals():
            doc.close()


def parse_text(file_path: Path, corpus_root: Path) -> Optional[Document]:
    """
    Parse a plain text file.

    Args:
        file_path: Path to text file
        corpus_root: Root directory of corpus

    Returns:
        Document object or None if parsing fails
    """
    try:
        # Read file with UTF-8 encoding
        raw_text = file_path.read_text(encoding="utf-8")

        # Normalize text
        normalized_text = normalize_text(raw_text)

        # Get relative path
        rel_path = str(file_path.relative_to(corpus_root))

        # Generate document UID (stable) and ID (versioned)
        doc_uid = generate_doc_uid(rel_path)
        normalized_text_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        doc_id = generate_doc_id(doc_uid, normalized_text_hash)

        # Use filename as title
        title = file_path.stem

        return Document(
            doc_uid=doc_uid,
            doc_id=doc_id,
            source_path=rel_path,
            raw_text=raw_text,
            normalized_text=normalized_text,
            title=title,
            pages=None,  # Text files don't have pages
            extracted_at=datetime.now(),
            metadata={},
        )

    except Exception as e:
        # Log error and return None
        print(f"Error parsing text file {file_path}: {e}")
        return None
