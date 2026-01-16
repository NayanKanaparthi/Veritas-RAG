"""
Deterministic ID generation for Veritas RAG.

ID Policy (Deterministic Hashes):
- doc_uid: sha256(rel_path)[:16] - stable across content changes
- doc_id: sha256(doc_uid + normalized_text_hash)[:16] - versioned by content
- chunk_id: hash of doc_uid + offset_start + offset_end + chunk_text_hash
  (uses doc_uid for stability across content changes, and stable offsets for
  position stability. This ensures chunk IDs remain stable even when document
  content changes, as long as the chunk text and offsets remain the same)
"""

import hashlib
import os
from pathlib import Path


def normalize_path(rel_path: str) -> str:
    """
    Normalize relative path from corpus root.

    - Use forward slashes
    - Resolve . and ..
    - Ensure consistent representation
    """
    # Normalize path separators
    normalized = rel_path.replace("\\", "/")
    # Resolve . and ..
    parts = []
    for part in normalized.split("/"):
        if part == ".":
            continue
        elif part == "..":
            if parts:
                parts.pop()
        else:
            parts.append(part)
    return "/".join(parts)


def generate_doc_uid(rel_path: str) -> str:
    """
    Generate stable document UID (stable across content changes).

    Args:
        rel_path: Relative path from corpus root (NOT absolute path)

    Returns:
        16-character hex digest of the path hash
    """
    normalized_path = normalize_path(rel_path)
    hash_obj = hashlib.sha256(normalized_path.encode("utf-8"))
    return hash_obj.hexdigest()[:16]


def generate_doc_id(doc_uid: str, normalized_text_hash: str) -> str:
    """
    Generate versioned document ID (changes with content).

    Args:
        doc_uid: Stable document UID
        normalized_text_hash: SHA256 hash of normalized text (Unicode NFKC)

    Returns:
        16-character hex digest of the combined hash
    """
    # Combine doc_uid and content hash
    combined = f"{doc_uid}{normalized_text_hash}"
    # Generate hash
    hash_obj = hashlib.sha256(combined.encode("utf-8"))
    return hash_obj.hexdigest()[:16]


def generate_chunk_id(
    doc_uid: str, offset_start: int, offset_end: int, chunk_text: str
) -> str:
    """
    Generate deterministic chunk ID.

    Uses doc_uid (stable across content changes) and stable offsets for ID stability.
    This ensures chunk IDs remain stable even when document content changes, as long
    as the chunk text and offsets remain the same.

    Args:
        doc_uid: Stable document UID (sha256(rel_path)[:16])
        offset_start: Start offset in Document.normalized_text (inclusive)
        offset_end: End offset in Document.normalized_text (exclusive)
        chunk_text: The chunk text content

    Returns:
        16-character hex digest of the combined hash
    """
    # Hash the chunk text
    chunk_text_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
    # Combine doc_uid (not doc_id), offsets, and text hash
    combined = f"{doc_uid}{offset_start}{offset_end}{chunk_text_hash}"
    # Generate hash
    hash_obj = hashlib.sha256(combined.encode("utf-8"))
    return hash_obj.hexdigest()[:16]
