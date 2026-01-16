"""
Document ingestion pipeline: parsing, normalization, and chunking.
"""

from veritas_rag.ingestion.chunker import FixedSizeChunker
from veritas_rag.ingestion.normalizer import normalize_text
from veritas_rag.ingestion.parsers import parse_document, parse_pdf, parse_text

__all__ = [
    "parse_document",
    "parse_pdf",
    "parse_text",
    "normalize_text",
    "FixedSizeChunker",
]
