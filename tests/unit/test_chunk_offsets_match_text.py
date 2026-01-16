"""Test that chunk offsets match stored text exactly."""

import pytest
from veritas_rag.core.contracts import Document, Config
from veritas_rag.ingestion.chunker import FixedSizeChunker
from veritas_rag.core.ids import generate_doc_uid, generate_doc_id


def test_chunk_text_matches_offset_slice():
    """Critical invariant: chunk.text must equal normalized_text[offset_start:offset_end]."""
    config = Config(chunk_size=100, chunk_overlap=10)
    chunker = FixedSizeChunker(config)

    # Create a test document
    normalized_text = "This is a test document. " * 10
    doc_uid = generate_doc_uid("test.txt")
    doc_id = generate_doc_id(doc_uid, "hash")

    doc = Document(
        doc_uid=doc_uid,
        doc_id=doc_id,
        source_path="test.txt",
        raw_text=normalized_text,
        normalized_text=normalized_text,
    )

    # Chunk the document
    chunks = chunker.chunk_document(doc)

    # Verify invariant for all chunks
    for chunk in chunks:
        expected_text = doc.normalized_text[chunk.offset_start:chunk.offset_end]
        assert chunk.text == expected_text, (
            f"Chunk {chunk.chunk_id}: text mismatch. "
            f"Expected: {repr(expected_text)}, Got: {repr(chunk.text)}"
        )
