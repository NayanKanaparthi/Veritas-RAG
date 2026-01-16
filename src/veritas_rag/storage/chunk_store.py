"""
Binary chunk store for Veritas RAG.

Stores chunks in compressed binary format with integrity checksums.
"""

import json
import struct
from pathlib import Path
from typing import Dict, List, Optional

import xxhash

from veritas_rag.core.contracts import Chunk, SourceRef
from veritas_rag.storage.compression import compress_data, decompress_data


class ChunkStore:
    """
    Binary chunk store with compression and checksums.

    Format:
    - chunks.bin: Append-only binary file with compressed chunk payloads
    - chunks.idx: Index file with explicit record schema
    - docs.meta: JSON metadata for documents
    """

    # Index record format (binary struct)
    # Fields (12 total):
    # chunk_id (32 bytes), doc_uid (32 bytes), doc_id (32 bytes),
    # store_offset (Q, 8 bytes), length (I, 4 bytes), checksum (I, 4 bytes unsigned),
    # is_active (B, 1 byte), offset_start (Q, 8 bytes), offset_end (Q, 8 bytes),
    # chunk_index (I, 4 bytes unsigned), page_start (i, 4 bytes signed, -1 if None),
    # page_end (i, 4 bytes signed, -1 if None)
    # Use explicit little-endian format with no padding
    INDEX_RECORD_FORMAT = "<32s32s32sQIIBQQIii"
    INDEX_RECORD_SIZE = struct.calcsize(INDEX_RECORD_FORMAT)

    def __init__(self, store_dir: Path):
        """
        Initialize chunk store.

        Args:
            store_dir: Directory containing chunks.bin, chunks.idx, docs.meta
        """
        self.store_dir = store_dir
        self.chunks_bin_path = store_dir / "chunks.bin"
        self.chunks_idx_path = store_dir / "chunks.idx"
        self.docs_meta_path = store_dir / "docs.meta"

        # In-memory index for fast lookups
        self.index: Dict[str, dict] = {}  # chunk_id -> record dict
        self.doc_uid_to_chunks: Dict[str, List[str]] = {}  # doc_uid -> list of chunk_ids
        self.doc_id_to_source_path: Dict[str, str] = {}  # doc_id -> source_path (for citations)

    def write_chunk(self, chunk: Chunk, compressed_data: bytes, checksum: int):
        """
        Write chunk to store (append-only).

        Args:
            chunk: Chunk object
            compressed_data: Compressed chunk payload
            checksum: CRC32/xxhash checksum of uncompressed data
        """
        # Append to chunks.bin
        store_offset = self.chunks_bin_path.stat().st_size if self.chunks_bin_path.exists() else 0

        with open(self.chunks_bin_path, "ab") as f:
            f.write(compressed_data)

        length = len(compressed_data)

        # Write index record
        self._write_index_record(
            chunk_id=chunk.chunk_id,
            doc_uid=chunk.doc_uid,
            doc_id=chunk.doc_id,
            store_offset=store_offset,
            length=length,
            checksum=checksum,
            is_active=True,
            offset_start=chunk.offset_start,
            offset_end=chunk.offset_end,
            chunk_index=chunk.chunk_index,
            page_start=chunk.page_start if chunk.page_start is not None else -1,
            page_end=chunk.page_end if chunk.page_end is not None else -1,
        )

        # Update in-memory index
        self.index[chunk.chunk_id] = {
            "doc_uid": chunk.doc_uid,
            "doc_id": chunk.doc_id,
            "store_offset": store_offset,
            "length": length,
            "checksum": checksum,
            "is_active": True,
            "offset_start": chunk.offset_start,
            "offset_end": chunk.offset_end,
            "chunk_index": chunk.chunk_index,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
        }

        # Update doc_uid_to_chunks mapping
        if chunk.doc_uid not in self.doc_uid_to_chunks:
            self.doc_uid_to_chunks[chunk.doc_uid] = []
        self.doc_uid_to_chunks[chunk.doc_uid].append(chunk.chunk_id)

    def read_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """
        Read chunk from store.

        Args:
            chunk_id: Chunk ID

        Returns:
            Chunk object or None if not found or inactive
        """
        if chunk_id not in self.index:
            return None

        record = self.index[chunk_id]
        if not record["is_active"]:
            return None

        # Read compressed data
        with open(self.chunks_bin_path, "rb") as f:
            f.seek(record["store_offset"])
            compressed_data = f.read(record["length"])

        # Decompress
        decompressed_data = decompress_data(compressed_data)
        text = decompressed_data.decode("utf-8")

        # Validate checksum (xxhash32, stored as unsigned int)
        computed_checksum = xxhash.xxh32(decompressed_data).intdigest()
        if computed_checksum != record["checksum"]:
            raise ValueError(f"Checksum mismatch for chunk {chunk_id}")

        # Get source_path from doc_id mapping
        source_path = self.doc_id_to_source_path.get(record["doc_id"], "")

        source_ref = SourceRef(
            source_path=source_path,
            offset_start=record["offset_start"],
            offset_end=record["offset_end"],
            page_start=record["page_start"] if record["page_start"] != -1 else None,
            page_end=record["page_end"] if record["page_end"] != -1 else None,
            section_path=None,
        )

        chunk = Chunk(
            chunk_id=chunk_id,
            doc_uid=record["doc_uid"],
            doc_id=record["doc_id"],
            text=text,
            offset_start=record["offset_start"],
            offset_end=record["offset_end"],
            chunk_index=record["chunk_index"],
            source_ref=source_ref,
            page_start=record["page_start"] if record["page_start"] != -1 else None,
            page_end=record["page_end"] if record["page_end"] != -1 else None,
            section_path=None,
            metadata={},
        )

        return chunk

    def _write_index_record(
        self,
        chunk_id: str,
        doc_uid: str,
        doc_id: str,
        store_offset: int,
        length: int,
        checksum: int,
        is_active: bool,
        offset_start: int,
        offset_end: int,
        chunk_index: int,
        page_start: int,
        page_end: int,
    ):
        """Write a single index record to chunks.idx."""
        # Pad IDs to 32 bytes (16 hex chars)
        chunk_id_bytes = chunk_id.encode("utf-8")[:32].ljust(32, b"\0")
        doc_uid_bytes = doc_uid.encode("utf-8")[:32].ljust(32, b"\0")
        doc_id_bytes = doc_id.encode("utf-8")[:32].ljust(32, b"\0")

        # Pack record
        record = struct.pack(
            self.INDEX_RECORD_FORMAT,
            chunk_id_bytes,
            doc_uid_bytes,
            doc_id_bytes,
            store_offset,
            length,
            checksum,
            1 if is_active else 0,
            offset_start,
            offset_end,
            chunk_index,
            page_start,
            page_end,
        )

        # Append to index file
        with open(self.chunks_idx_path, "ab") as f:
            f.write(record)

    def load_index(self):
        """Load index from chunks.idx into memory."""
        if not self.chunks_idx_path.exists():
            return

        self.index = {}
        self.doc_uid_to_chunks = {}
        self.doc_id_to_source_path = {}

        # Load docs.meta to build doc_id -> source_path mapping
        docs_meta = self.load_docs_meta()
        for doc_uid, doc_info in docs_meta.items():
            if "doc_id" in doc_info and "source_path" in doc_info:
                self.doc_id_to_source_path[doc_info["doc_id"]] = doc_info["source_path"]

        with open(self.chunks_idx_path, "rb") as f:
            while True:
                record_bytes = f.read(self.INDEX_RECORD_SIZE)
                if len(record_bytes) < self.INDEX_RECORD_SIZE:
                    break

                # Unpack record
                (
                    chunk_id_bytes,
                    doc_uid_bytes,
                    doc_id_bytes,
                    store_offset,
                    length,
                    checksum,
                    is_active_byte,
                    offset_start,
                    offset_end,
                    chunk_index,
                    page_start,
                    page_end,
                ) = struct.unpack(self.INDEX_RECORD_FORMAT, record_bytes)

                chunk_id = chunk_id_bytes.rstrip(b"\0").decode("utf-8")
                doc_uid = doc_uid_bytes.rstrip(b"\0").decode("utf-8")
                doc_id = doc_id_bytes.rstrip(b"\0").decode("utf-8")
                is_active = bool(is_active_byte)

                # Last record wins: if we've seen this chunk_id before, overwrite
                # (this handles tombstones - last record determines active state)
                self.index[chunk_id] = {
                    "doc_uid": doc_uid,
                    "doc_id": doc_id,
                    "store_offset": store_offset,
                    "length": length,
                    "checksum": checksum,
                    "is_active": is_active,
                    "offset_start": offset_start,
                    "offset_end": offset_end,
                    "chunk_index": chunk_index,
                    "page_start": page_start if page_start != -1 else None,
                    "page_end": page_end if page_end != -1 else None,
                }

                if doc_uid not in self.doc_uid_to_chunks:
                    self.doc_uid_to_chunks[doc_uid] = []
                if chunk_id not in self.doc_uid_to_chunks[doc_uid]:
                    self.doc_uid_to_chunks[doc_uid].append(chunk_id)

    def tombstone_chunk(self, chunk_id: str):
        """
        Tombstone a chunk by appending an inactive record (append-only design).

        Args:
            chunk_id: Chunk ID to tombstone
        """
        if chunk_id not in self.index:
            return

        record = self.index[chunk_id]
        # Append new record with is_active=False
        self._write_index_record(
            chunk_id=chunk_id,
            doc_uid=record["doc_uid"],
            doc_id=record["doc_id"],
            store_offset=record["store_offset"],  # Same offset (tombstone marker)
            length=record["length"],
            checksum=record["checksum"],
            is_active=False,
            offset_start=record["offset_start"],
            offset_end=record["offset_end"],
            chunk_index=record["chunk_index"],
            page_start=record["page_start"] if record["page_start"] is not None else -1,
            page_end=record["page_end"] if record["page_end"] is not None else -1,
        )

        # Update in-memory index
        self.index[chunk_id]["is_active"] = False

    def tombstone_document(self, doc_uid: str):
        """
        Tombstone all chunks for a document.

        Args:
            doc_uid: Stable document UID
        """
        if doc_uid not in self.doc_uid_to_chunks:
            return

        # Tombstone each chunk
        for chunk_id in self.doc_uid_to_chunks[doc_uid]:
            self.tombstone_chunk(chunk_id)

    def save_docs_meta(self, docs_meta: Dict):
        """Save document metadata to docs.meta."""
        with open(self.docs_meta_path, "w") as f:
            json.dump(docs_meta, f, indent=2)

    def load_docs_meta(self) -> Dict:
        """Load document metadata from docs.meta."""
        if not self.docs_meta_path.exists():
            return {}

        with open(self.docs_meta_path, "r") as f:
            return json.load(f)

    def validate_invariants(self) -> List[str]:
        """
        Validate chunk store invariants.

        Returns:
            List of error messages (empty if all valid)
        """
        errors = []

        # Check chunks.bin file size
        if self.chunks_bin_path.exists():
            bin_size = self.chunks_bin_path.stat().st_size

            # Check all index records point to valid offsets
            for chunk_id, record in self.index.items():
                if not record["is_active"]:
                    continue
                store_offset = record["store_offset"]  # Use existing field name
                length = record["length"]
                if store_offset + length > bin_size:
                    errors.append(
                        f"Chunk {chunk_id}: store_offset {store_offset} + length {length} "
                        f"exceeds chunks.bin size {bin_size}"
                    )

        # Verify tombstoned chunks aren't in active index
        # (load_index already enforces "last record wins", but double-check)
        for chunk_id, record in self.index.items():
            if not record["is_active"]:
                # Should not be returned by any API
                # This is already enforced, but document it here
                pass

        return errors
