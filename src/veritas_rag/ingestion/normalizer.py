"""
Text normalization for Veritas RAG.

Normalizes text using Unicode NFKC while preserving punctuation important
for code and IDs.
"""

import unicodedata


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing.

    - Unicode normalization (NFKC)
    - Whitespace normalization (preserves newlines as delimiters)
    - Preserves punctuation for code/IDs

    Args:
        text: Raw text to normalize

    Returns:
        Normalized text (Unicode NFKC)
    """
    # Unicode NFKC normalization
    normalized = unicodedata.normalize("NFKC", text)

    # Whitespace normalization: collapse spaces/tabs, but preserve newlines
    # This ensures page boundaries remain stable in normalized_text
    import re

    # Replace multiple spaces/tabs (but not newlines) with single space
    normalized = re.sub(r"[ \t]+", " ", normalized)
    # Normalize newlines: convert \r\n to \n, then ensure consistent newline usage
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    # Strip leading/trailing whitespace
    normalized = normalized.strip()

    return normalized
