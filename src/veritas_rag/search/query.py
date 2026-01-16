"""
Query processing and explanation for Veritas RAG.
"""

from typing import List

from veritas_rag.core.contracts import RetrievalResult, SourceRef
from veritas_rag.search.bm25_index import BM25Index
from veritas_rag.search.tokenizer import BM25Tokenizer


class QueryProcessor:
    """Processes queries and generates explainable results."""

    def __init__(self, index: BM25Index, chunk_store=None):
        """
        Initialize query processor.

        Args:
            index: BM25Index instance
            chunk_store: ChunkStore instance (optional, for snippets)
        """
        self.index = index
        self.tokenizer = index.tokenizer
        self.chunk_store = chunk_store

    def process_query(
        self, query: str, top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Process query and return explainable results.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects with explanations
        """
        # Get search results
        results = self.index.search(query, top_k=top_k)

        # Tokenize query to find matched terms
        query_tokens = set(self.tokenizer.tokenize_query(query))

        # Build RetrievalResult objects
        retrieval_results = []
        for chunk_id, score in results:
            matched_terms = []
            snippet = ""
            source_ref = SourceRef(
                source_path="",
                offset_start=0,
                offset_end=0,
                page_start=None,
                page_end=None,
                section_path=None,
            )

            # If we have chunk_store, fetch chunk to get real data
            if self.chunk_store:
                chunk = self.chunk_store.read_chunk(chunk_id)
                if chunk:
                    # Find matched terms: intersection of query tokens and chunk tokens
                    chunk_tokens = set(self.tokenizer.tokenize(chunk.text))
                    matched_terms = sorted(list(query_tokens & chunk_tokens))

                    # Generate snippet: find first matched term and return ~200 chars around it
                    snippet = self._generate_snippet(chunk.text, matched_terms)

                    source_ref = chunk.source_ref

            retrieval_result = RetrievalResult(
                chunk_id=chunk_id,
                score=score,
                matched_terms=matched_terms,
                snippet=snippet,
                source_ref=source_ref,
            )

            retrieval_results.append(retrieval_result)

        return retrieval_results

    def _generate_snippet(self, text: str, matched_terms: List[str], max_length: int = 200) -> str:
        """
        Generate snippet around first matched term.

        Args:
            text: Chunk text
            matched_terms: List of matched terms
            max_length: Maximum snippet length

        Returns:
            Snippet string
        """
        if not matched_terms or not text:
            return text[:max_length] if text else ""

        # Find first occurrence of any matched term (case-insensitive)
        text_lower = text.lower()
        first_pos = len(text)

        for term in matched_terms:
            pos = text_lower.find(term.lower())
            if pos != -1 and pos < first_pos:
                first_pos = pos

        if first_pos == len(text):
            # No match found, return start of text
            return text[:max_length]

        # Extract snippet around first match
        start = max(0, first_pos - max_length // 2)
        end = min(len(text), first_pos + max_length // 2)

        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        return snippet
