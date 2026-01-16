"""
Retrieval and fetch pipeline for Veritas RAG.
"""

from typing import List, Tuple

from veritas_rag.core.contracts import Chunk, RetrievalResult
from veritas_rag.search.bm25_index import BM25Index
from veritas_rag.search.query import QueryProcessor
from veritas_rag.storage.chunk_store import ChunkStore


class RetrievalPipeline:
    """
    End-to-end retrieval pipeline: query → retrieve → fetch → return chunks.
    """

    def __init__(self, index: BM25Index, chunk_store: ChunkStore):
        """
        Initialize retrieval pipeline.

        Args:
            index: BM25Index instance
            chunk_store: ChunkStore instance
        """
        self.index = index
        self.chunk_store = chunk_store
        self.query_processor = QueryProcessor(index, chunk_store)

    def retrieve_ids(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve top-k chunk IDs and scores only (no disk reads).

        Args:
            query: Query string
            top_k: Number of results

        Returns:
            List of (chunk_id, score) tuples, sorted by score descending
        """
        return self.index.search(query, top_k)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve top-k chunks for a query.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects
        """
        return self.query_processor.process_query(query, top_k)

    def fetch_chunks(self, chunk_ids: List[str]) -> List[Chunk]:
        """
        Fetch chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs

        Returns:
            List of Chunk objects
        """
        chunks = []
        for chunk_id in chunk_ids:
            chunk = self.chunk_store.read_chunk(chunk_id)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    def retrieve_and_fetch(
        self, query: str, top_k: int = 10
    ) -> tuple[List[RetrievalResult], List[Chunk]]:
        """
        Retrieve and fetch chunks in one call.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            Tuple of (RetrievalResult list, Chunk list)
        """
        results = self.retrieve(query, top_k)
        chunk_ids = [r.chunk_id for r in results]
        chunks = self.fetch_chunks(chunk_ids)
        return results, chunks
