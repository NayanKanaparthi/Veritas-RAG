"""Test score normalization for CLI display."""

from veritas_rag.cli.main import normalize_scores_for_display
from veritas_rag.core.contracts import RetrievalResult, SourceRef


def test_normalize_scores_never_negative():
    """Verify normalization never produces negative values."""
    # Test with all negative scores
    source_ref = SourceRef(source_path="test.txt", offset_start=0, offset_end=10)
    results = [
        RetrievalResult(chunk_id="1", score=-5.0, matched_terms=["term"], snippet="snippet", source_ref=source_ref),
        RetrievalResult(chunk_id="2", score=-3.0, matched_terms=["term"], snippet="snippet", source_ref=source_ref),
        RetrievalResult(chunk_id="3", score=-1.0, matched_terms=["term"], snippet="snippet", source_ref=source_ref),
    ]
    
    display_scores = normalize_scores_for_display(results)
    assert all(score >= 0 for score in display_scores)
    assert display_scores[0] == 0.0  # Min score shifted to 0
    assert display_scores[1] == 2.0
    assert display_scores[2] == 4.0


def test_normalize_scores_preserves_order():
    """Verify normalization preserves ranking order."""
    # Test with mixed scores
    source_ref = SourceRef(source_path="test.txt", offset_start=0, offset_end=10)
    results = [
        RetrievalResult(chunk_id="1", score=-2.0, matched_terms=["term"], snippet="snippet", source_ref=source_ref),
        RetrievalResult(chunk_id="2", score=0.0, matched_terms=["term"], snippet="snippet", source_ref=source_ref),
        RetrievalResult(chunk_id="3", score=5.0, matched_terms=["term"], snippet="snippet", source_ref=source_ref),
    ]
    
    display_scores = normalize_scores_for_display(results)
    
    # Order should be preserved
    assert display_scores[0] < display_scores[1] < display_scores[2]
    assert display_scores[0] == 0.0  # Min score shifted to 0
    assert display_scores[1] == 2.0
    assert display_scores[2] == 7.0


def test_normalize_scores_all_positive():
    """Verify normalization works with all positive scores."""
    source_ref = SourceRef(source_path="test.txt", offset_start=0, offset_end=10)
    results = [
        RetrievalResult(chunk_id="1", score=1.0, matched_terms=["term"], snippet="snippet", source_ref=source_ref),
        RetrievalResult(chunk_id="2", score=2.0, matched_terms=["term"], snippet="snippet", source_ref=source_ref),
        RetrievalResult(chunk_id="3", score=3.0, matched_terms=["term"], snippet="snippet", source_ref=source_ref),
    ]
    
    display_scores = normalize_scores_for_display(results)
    
    # All positive scores should remain unchanged (shift = 0)
    assert display_scores[0] == 1.0
    assert display_scores[1] == 2.0
    assert display_scores[2] == 3.0


def test_normalize_scores_empty():
    """Verify normalization handles empty results."""
    results = []
    display_scores = normalize_scores_for_display(results)
    assert display_scores == []
