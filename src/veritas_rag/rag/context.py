"""
Context assembly for RAG (MVP stub - no LLM).
"""

from typing import List

from veritas_rag.core.contracts import Chunk


def assemble_context(chunks: List[Chunk]) -> str:
    """
    Assemble context from chunks with citations (MVP stub).

    Args:
        chunks: List of Chunk objects

    Returns:
        Context string with citations
    """
    context_parts = []
    seen_docs = set()

    for chunk in chunks:
        # Deduplicate by doc/section (simplified for MVP)
        doc_key = chunk.doc_id
        if doc_key in seen_docs:
            continue
        seen_docs.add(doc_key)

        # Format citation
        citation = f"[Doc: {chunk.source_ref.source_path}"
        if chunk.source_ref.page_start is not None:
            citation += f", Page: {chunk.source_ref.page_start}"
        citation += "]"

        context_parts.append(f"{citation} {chunk.text}")

    return "\n\n".join(context_parts)
