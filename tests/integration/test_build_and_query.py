"""
Integration test: build artifact from corpus and query it.
"""

import tempfile
from pathlib import Path

from veritas_rag import build_artifact, load_artifact
from veritas_rag.core import Config


def test_build_and_query():
    """Test end-to-end: build artifact from tiny corpus and query it."""
    # Create temporary corpus
    with tempfile.TemporaryDirectory() as temp_dir:
        corpus_dir = Path(temp_dir) / "corpus"
        corpus_dir.mkdir()
        artifact_dir = Path(temp_dir) / "artifact"

        # Create a simple text file
        test_file = corpus_dir / "test.txt"
        test_file.write_text(
            "This is a test document about machine learning. "
            "Machine learning is a subset of artificial intelligence. "
            "It involves training models on data."
        )

        # Build artifact
        config = Config(chunk_size=20, chunk_overlap=5)
        build_artifact(str(corpus_dir), str(artifact_dir), config)

        # Verify artifact was created
        assert (artifact_dir / "manifest.json").exists()
        assert (artifact_dir / "chunks.bin").exists()
        assert (artifact_dir / "chunks.idx").exists()
        assert (artifact_dir / "bm25_index.pkl").exists()
        assert (artifact_dir / "docs.meta").exists()

        # Load and query
        artifact = load_artifact(str(artifact_dir))
        results = artifact.retrieve("machine learning", top_k=5)

        # Should get at least one result
        assert len(results) > 0
        assert results[0].chunk_id is not None
        assert len(results[0].matched_terms) > 0  # Verify matched_terms contains query terms
        # Verify query terms appear in matched_terms
        query_terms = {"machine", "learning"}
        assert any(term in query_terms for term in results[0].matched_terms)

        # Fetch chunks
        chunk_ids = [r.chunk_id for r in results]
        chunks = artifact.fetch_chunks(chunk_ids)

        # Verify chunks have source paths
        assert len(chunks) > 0
        assert chunks[0].source_ref.source_path != ""

        # Test retrieve_ids
        ids_results = artifact.retrieve_ids("machine learning", top_k=5)
        assert len(ids_results) > 0
        assert isinstance(ids_results[0], tuple)
        assert len(ids_results[0]) == 2  # (chunk_id, score)
