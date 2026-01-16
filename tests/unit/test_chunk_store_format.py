"""Test that ChunkStore index record format matches pack/unpack exactly."""

import struct

import pytest

from veritas_rag.storage.chunk_store import ChunkStore


def test_index_record_format():
    """Verify INDEX_RECORD_FORMAT matches pack/unpack exactly."""
    # Test data matching actual usage
    chunk_id = "a" * 16  # 16 hex chars = 32 bytes when encoded
    doc_uid = "b" * 16
    doc_id = "c" * 16
    store_offset = 1000
    length = 500
    checksum = 12345
    is_active = 1
    offset_start = 0
    offset_end = 100
    chunk_index = 0
    page_start = -1
    page_end = -1

    # Pack (matching ChunkStore._write_index_record logic)
    chunk_id_bytes = chunk_id.encode("utf-8")[:32].ljust(32, b"\0")
    doc_uid_bytes = doc_uid.encode("utf-8")[:32].ljust(32, b"\0")
    doc_id_bytes = doc_id.encode("utf-8")[:32].ljust(32, b"\0")

    packed = struct.pack(
        ChunkStore.INDEX_RECORD_FORMAT,
        chunk_id_bytes,
        doc_uid_bytes,
        doc_id_bytes,
        store_offset,
        length,
        checksum,
        is_active,
        offset_start,
        offset_end,
        chunk_index,
        page_start,
        page_end,
    )

    # Verify packed size matches expected
    assert len(packed) == ChunkStore.INDEX_RECORD_SIZE

    # Unpack
    unpacked = struct.unpack(ChunkStore.INDEX_RECORD_FORMAT, packed)
    assert len(unpacked) == 12

    # Verify values match
    assert unpacked[0] == chunk_id_bytes
    assert unpacked[1] == doc_uid_bytes
    assert unpacked[2] == doc_id_bytes
    assert unpacked[3] == store_offset
    assert unpacked[4] == length
    assert unpacked[5] == checksum
    assert unpacked[6] == is_active
    assert unpacked[7] == offset_start
    assert unpacked[8] == offset_end
    assert unpacked[9] == chunk_index
    assert unpacked[10] == page_start
    assert unpacked[11] == page_end
