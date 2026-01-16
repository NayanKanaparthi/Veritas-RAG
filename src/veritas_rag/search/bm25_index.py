"""
BM25 sparse retrieval index for Veritas RAG.

MVP: Uses rank-bm25 library with pickle persistence (up to 100k chunks).
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from veritas_rag.core.contracts import Chunk, Config
from veritas_rag.search.tokenizer import BM25Tokenizer


class BM25Index:
    """
    BM25 sparse retrieval index.

    MVP: Pickle-based persistence for simplicity.
    Phase 2: Replace with persistent inverted index format for scale.
    MVP Scale: Up to 100k chunks (document this limit).
    """

    def __init__(self, config: Config):
        """
        Initialize BM25 index.

        Args:
            config: Configuration with BM25 parameters
        """
        if BM25Okapi is None:
            raise ImportError("rank-bm25 is required for BM25 indexing")

        self.config = config
        self.tokenizer = BM25Tokenizer(use_stopwords=config.bm25_use_stopwords)
        self.bm25: BM25Okapi = None
        self.chunk_ids: List[str] = []
        self.chunk_id_to_index: Dict[str, int] = {}

    def build(self, chunks: List[Chunk]):
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of Chunk objects to index
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list")

        # Tokenize all chunks
        tokenized_corpus = []
        self.chunk_ids = []
        self.chunk_id_to_index = {}

        for idx, chunk in enumerate(chunks):
            tokens = self.tokenizer.tokenize(chunk.text)
            tokenized_corpus.append(tokens)
            self.chunk_ids.append(chunk.chunk_id)
            self.chunk_id_to_index[chunk.chunk_id] = idx

        # Build BM25 index
        self.bm25 = BM25Okapi(
            tokenized_corpus,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search index with query.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of (chunk_id, score) tuples, sorted by score descending
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build() first.")

        # Tokenize query
        query_tokens = self.tokenizer.tokenize_query(query)

        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k results
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        # Return chunk_ids with scores
        results = [(self.chunk_ids[idx], float(scores[idx])) for idx in top_indices]
        return results

    def save(self, file_path: Path):
        """
        Save index to disk (pickle format for MVP).

        Args:
            file_path: Path to save index
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build() first.")

        data = {
            "bm25": self.bm25,
            "chunk_ids": self.chunk_ids,
            "chunk_id_to_index": self.chunk_id_to_index,
            "config": self.config,
        }

        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, file_path: Path, unsafe_load: bool = False) -> "BM25Index":
        """
        Load index from disk.

        Args:
            file_path: Path to index file
            unsafe_load: If False, print warning about pickle safety (default: False)

        Returns:
            Loaded BM25Index instance
        """
        if not unsafe_load:
            import warnings
            warnings.warn(
                "Loading BM25 index from pickle file. "
                "Do not load artifacts from untrusted sources; pickle can execute code.",
                UserWarning
            )

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        index = cls(data["config"])
        index.bm25 = data["bm25"]
        index.chunk_ids = data["chunk_ids"]
        index.chunk_id_to_index = data["chunk_id_to_index"]

        return index
