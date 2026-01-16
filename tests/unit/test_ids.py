"""
Tests for ID generation.
"""

import hashlib

from veritas_rag.core.ids import generate_chunk_id, generate_doc_id, generate_doc_uid


def test_generate_doc_id():
    """Test deterministic doc_id generation."""
    rel_path = "documents/test.txt"
    doc_uid = generate_doc_uid(rel_path)
    normalized_text = "Hello world"
    normalized_text_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()

    doc_id1 = generate_doc_id(doc_uid, normalized_text_hash)
    doc_id2 = generate_doc_id(doc_uid, normalized_text_hash)

    # Should be deterministic
    assert doc_id1 == doc_id2
    assert len(doc_id1) == 16


def test_generate_chunk_id():
    """Test deterministic chunk_id generation."""
    doc_uid = "abc123def456"  # Stable UID
    offset_start = 0
    offset_end = 10
    chunk_text = "Hello world"

    chunk_id1 = generate_chunk_id(doc_uid, offset_start, offset_end, chunk_text)
    chunk_id2 = generate_chunk_id(doc_uid, offset_start, offset_end, chunk_text)

    # Should be deterministic
    assert chunk_id1 == chunk_id2
    assert len(chunk_id1) == 16
