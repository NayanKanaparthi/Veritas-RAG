"""
BM25 tokenizer for Veritas RAG.

BM25 tokenization must be term-based (regex/word tokenizer), NOT LLM tokens.
Do NOT use tiktoken for BM25 tokenization.
"""

import re
from typing import List, Optional, Set


class BM25Tokenizer:
    """
    Word-based tokenizer for BM25.

    Uses regex word splits, lowercase normalization, and optional stopwords.
    """

    def __init__(self, use_stopwords: bool = False, stopwords: Optional[Set[str]] = None):
        """
        Initialize tokenizer.

        Args:
            use_stopwords: Whether to filter stopwords
            stopwords: Custom stopword set (if None and use_stopwords=True, uses default)
        """
        self.use_stopwords = use_stopwords
        if stopwords is None:
            # Default English stopwords (minimal set for MVP)
            self.stopwords = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "should",
                "could",
                "may",
                "might",
                "must",
                "can",
            }
        else:
            self.stopwords = stopwords

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.

        - Lowercase
        - Regex word splits (\\w+)
        - Optional stopword filtering

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Lowercase
        text_lower = text.lower()

        # Extract words using regex \w+ (word characters: letters, digits, underscore)
        tokens = re.findall(r"\w+", text_lower)

        # Filter stopwords if enabled
        if self.use_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        return tokens

    def tokenize_query(self, query: str) -> List[str]:
        """
        Tokenize a query (same as tokenize, but kept separate for clarity).

        Args:
            query: Query string

        Returns:
            List of tokens
        """
        return self.tokenize(query)
