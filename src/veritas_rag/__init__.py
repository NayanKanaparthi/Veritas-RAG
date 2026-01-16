"""
Veritas RAG - Local-first RAG memory engine with sparse retrieval.
"""

from veritas_rag.artifact import Artifact, build_artifact, load_artifact
from veritas_rag.core import Config

__version__ = "0.1.0"

__all__ = [
    "build_artifact",
    "load_artifact",
    "Artifact",
    "Config",
]
