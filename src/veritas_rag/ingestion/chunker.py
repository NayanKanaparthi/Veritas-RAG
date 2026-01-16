"""
Chunking strategies for Veritas RAG.

MVP: Fixed-size chunking with overlap only.
"""

import re
from typing import List, Optional

from veritas_rag.core.contracts import Chunk, Config, Document, Page, SourceRef
from veritas_rag.core.ids import generate_chunk_id


def count_words(text: str) -> int:
    """Count words in text (simple whitespace split)."""
    return len(text.split())


class FixedSizeChunker:
    """Fixed-size chunking with overlap (MVP only)."""

    def __init__(self, config: Config):
        """
        Initialize chunker with configuration.

        Args:
            config: Configuration object with chunk_size, chunk_overlap, etc.
        """
        self.config = config
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Chunk a document into fixed-size chunks with overlap.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        normalized_text = document.normalized_text
        chunks = []

        # Simple word-based chunking (MVP)
        words = normalized_text.split()
        if not words:
            return chunks

        # Calculate chunk size in characters (approximate)
        # We'll use a sliding window approach
        current_pos = 0
        chunk_index = 0

        while current_pos < len(normalized_text):
            # Find the end position for this chunk
            # We want approximately chunk_size words
            end_pos = self._find_chunk_end(normalized_text, current_pos, self.chunk_size)

            if end_pos <= current_pos:
                # Can't make progress, break
                break

            # Extract chunk text (exact slice, no stripping)
            chunk_text = normalized_text[current_pos:end_pos]
            if not chunk_text:
                break

            # Calculate offsets
            offset_start = current_pos
            offset_end = end_pos

            # Derive page range for PDFs
            page_start, page_end = self._derive_page_range(
                document.pages, offset_start, offset_end
            )

            # Generate chunk ID (using doc_uid for stability across content changes)
            chunk_id = generate_chunk_id(
                document.doc_uid, offset_start, offset_end, chunk_text
            )

            # Create source reference (field order: required fields first)
            source_ref = SourceRef(
                source_path=document.source_path,
                offset_start=offset_start,
                offset_end=offset_end,
                page_start=page_start,
                page_end=page_end,
                section_path=None,  # MVP: no section paths
            )

            # Create chunk (field order: required fields first, then optional with defaults)
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_uid=document.doc_uid,
                doc_id=document.doc_id,
                text=chunk_text,
                offset_start=offset_start,
                offset_end=offset_end,
                chunk_index=chunk_index,
                source_ref=source_ref,
                page_start=page_start,
                page_end=page_end,
                section_path=None,
                metadata={},
            )

            chunks.append(chunk)
            chunk_index += 1

            # Move to next chunk position with overlap
            if end_pos >= len(normalized_text):
                break

            # Calculate overlap position (go back by overlap words)
            overlap_pos = self._find_chunk_start(
                normalized_text, end_pos, self.chunk_overlap
            )
            current_pos = overlap_pos

        return chunks

    def _find_chunk_end(self, text: str, start_pos: int, target_words: int) -> int:
        """Find the end position for a chunk of approximately target_words words."""
        words = text[start_pos:].split()
        if len(words) <= target_words:
            # Return end of text
            return len(text)

        # Find position after target_words words
        word_count = 0
        pos = start_pos
        while pos < len(text) and word_count < target_words:
            # Skip whitespace
            while pos < len(text) and text[pos].isspace():
                pos += 1
            # Skip word
            while pos < len(text) and not text[pos].isspace():
                pos += 1
            word_count += 1

        return pos

    def _find_chunk_start(self, text: str, end_pos: int, target_words: int) -> int:
        """Find the start position going backwards for overlap."""
        # Go backwards from end_pos to find start of overlap_words words back
        words = text[:end_pos].split()
        if len(words) <= target_words:
            return 0

        # Find position before target_words words from end
        word_count = 0
        pos = end_pos
        while pos > 0 and word_count < target_words:
            # Skip whitespace backwards
            while pos > 0 and text[pos - 1].isspace():
                pos -= 1
            # Skip word backwards
            while pos > 0 and not text[pos - 1].isspace():
                pos -= 1
            word_count += 1

        return pos

    def _derive_page_range(
        self, pages: Optional[List[Page]], offset_start: int, offset_end: int
    ) -> tuple[Optional[int], Optional[int]]:
        """
        Derive page_start/page_end from Page objects that overlap with chunk offsets.

        Args:
            pages: List of Page objects (may be None)
            offset_start: Chunk start offset (inclusive)
            offset_end: Chunk end offset (exclusive)

        Returns:
            Tuple of (page_start, page_end) or (None, None) if no pages
        """
        if not pages:
            return (None, None)

        page_start = None
        page_end = None

        for page in pages:
            # Check if page overlaps with chunk [offset_start, offset_end)
            # Page spans [page.offset_start, page.offset_end)
            page_overlaps = not (
                page.offset_end <= offset_start or page.offset_start >= offset_end
            )

            if page_overlaps:
                if page_start is None:
                    page_start = page.page_number
                page_end = page.page_number

        return (page_start, page_end)
