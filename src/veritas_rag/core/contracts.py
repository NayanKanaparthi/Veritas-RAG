"""
Core data structures (dataclasses) for Veritas RAG.

All core data structures are defined as explicit dataclasses.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional


@dataclass
class Config:
    """Configuration for artifact building and indexing."""

    # Chunking
    chunk_size: int = 512  # approximate word count or tokens
    chunk_overlap: int = 50
    use_llm_tokenizer: bool = False  # if True, use tiktoken; else word count

    # BM25 parameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_use_stopwords: bool = False

    # Compression
    compression: Literal["zstd"] = "zstd"  # MVP: "zstd" only; Phase 2: add "brotli"
    zstd_level: int = 3

    # Schema & versioning
    schema_version: str = "1.0"
    artifact_version: str = "1.0"

    # Indexing options
    enable_field_boosting: bool = False  # Phase 2
    enable_llm: bool = False  # Phase 2


@dataclass
class Page:
    """Represents a page from a PDF document (MVP: Optional)."""

    page_number: int
    text: str
    offset_start: int  # character offset in Document.normalized_text (inclusive)
    offset_end: int  # character offset in Document.normalized_text (exclusive)


@dataclass
class Document:
    """
    Represents a parsed document.

    Canonical Offset Reference: All offset_start/offset_end values refer to character
    positions in Document.normalized_text (Python slice convention: inclusive start,
    exclusive end).
    """

    doc_uid: str  # stable UID (sha256(rel_path)[:16]) - stable across content changes
    doc_id: str  # versioned ID (sha256(doc_uid + normalized_text_hash)[:16])
    source_path: str  # relative path from corpus root
    raw_text: str  # original extracted text
    normalized_text: str  # Unicode NFKC normalized text (canonical for offsets)
    title: Optional[str] = None
    pages: Optional[List[Page]] = None  # for PDFs (MVP: may be None)
    extracted_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceRef:
    """Source reference for citations."""

    source_path: str  # relative path from corpus root
    offset_start: int  # character offset in Document.normalized_text (inclusive)
    offset_end: int  # character offset in Document.normalized_text (exclusive)
    page_start: Optional[int] = None  # for PDFs (inclusive)
    page_end: Optional[int] = None  # for PDFs (inclusive)
    section_path: Optional[List[str]] = None


@dataclass
class Chunk:
    """
    Represents a text chunk for retrieval.

    Offset Convention:
    - offset_start is inclusive, offset_end is exclusive (Python slice convention)
    - All offsets refer to character positions in Document.normalized_text

    PDF Page Range Mapping:
    - page_start/page_end are derived by finding which Page objects have offset
      spans that overlap with the chunk's [offset_start, offset_end) range
    - If chunk spans multiple pages, page_start is the first page number,
      page_end is the last page number (inclusive)
    """

    chunk_id: str  # deterministic hash (see ID policy)
    doc_uid: str  # stable document UID
    doc_id: str  # versioned document ID
    text: str
    offset_start: int  # character offset in Document.normalized_text (inclusive)
    offset_end: int  # character offset in Document.normalized_text (exclusive)
    chunk_index: int  # position within document (informational, not used in ID)
    source_ref: SourceRef
    page_start: Optional[int] = None  # for PDFs (inclusive, derived from Page.offset spans)
    page_end: Optional[int] = None  # for PDFs (inclusive, derived from Page.offset spans)
    section_path: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result from retrieval query."""

    chunk_id: str
    score: float
    matched_terms: List[str]  # top matching terms
    snippet: str  # highlighted snippet
    source_ref: SourceRef


@dataclass
class ArtifactManifest:
    """Manifest for artifact metadata and integrity."""

    schema_version: str
    artifact_version: str
    build_timestamp: datetime
    total_docs: int
    total_chunks: int
    index_type: str  # "bm25"
    compression: str  # "zstd"
    checksums: Dict[str, str]  # file-level checksums
