"""
Core contracts and ID generation for Veritas RAG.
"""

from veritas_rag.core.contracts import (
    ArtifactManifest,
    Chunk,
    Config,
    Document,
    Page,
    RetrievalResult,
    SourceRef,
)
from veritas_rag.core.ids import generate_chunk_id, generate_doc_id, generate_doc_uid

__all__ = [
    "Config",
    "Page",
    "Document",
    "SourceRef",
    "Chunk",
    "RetrievalResult",
    "ArtifactManifest",
    "generate_doc_uid",
    "generate_doc_id",
    "generate_chunk_id",
]
