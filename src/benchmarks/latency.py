"""
Latency benchmarks for Veritas RAG.

Measures:
- Retrieval-only latency (query → top-k IDs)
- Retrieval+fetch latency (query → chunk text)
- Cold start load time
"""

import statistics
import time
from pathlib import Path
from typing import List, Tuple

try:
    from veritas_rag import load_artifact
except ImportError:
    # Fallback for when running from src/benchmarks directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from veritas_rag import load_artifact


def measure_retrieval_latency(
    artifact, query: str, top_k: int = 10, iterations: int = 100, warmup: int = 10
) -> dict:
    """
    Measure retrieval-only latency (query → top-k IDs).

    Args:
        artifact: Loaded Artifact instance
        query: Query string
        top_k: Number of results
        iterations: Number of iterations to measure
        warmup: Number of warmup iterations

    Returns:
        Dict with P50, P95, min, max, mean (in milliseconds)
    """
    # Warmup
    for _ in range(warmup):
        artifact.retrieve_ids(query, top_k)  # Use retrieve_ids

    # Measure
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        artifact.retrieve_ids(query, top_k)  # Use retrieve_ids
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)

    times.sort()
    return {
        "p50": statistics.median(times),
        "p95": times[int(len(times) * 0.95)],
        "min": min(times),
        "max": max(times),
        "mean": statistics.mean(times),
    }


def measure_retrieval_fetch_latency(
    artifact, query: str, top_k: int = 10, iterations: int = 100, warmup: int = 10
) -> dict:
    """
    Measure retrieval+fetch latency (query → chunk text).

    Args:
        artifact: Loaded Artifact instance
        query: Query string
        top_k: Number of results
        iterations: Number of iterations to measure
        warmup: Number of warmup iterations

    Returns:
        Dict with P50, P95, min, max, mean (in milliseconds)
    """
    # Warmup
    for _ in range(warmup):
        results = artifact.retrieve(query, top_k)
        chunk_ids = [r.chunk_id for r in results]
        artifact.fetch_chunks(chunk_ids)

    # Measure
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        results = artifact.retrieve(query, top_k)
        chunk_ids = [r.chunk_id for r in results]
        artifact.fetch_chunks(chunk_ids)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)

    times.sort()
    return {
        "p50": statistics.median(times),
        "p95": times[int(len(times) * 0.95)],
        "min": min(times),
        "max": max(times),
        "mean": statistics.mean(times),
    }


def measure_cold_start_time(artifact_path: str) -> float:
    """
    Measure cold start time (load artifact from disk).

    Args:
        artifact_path: Path to artifact directory

    Returns:
        Load time in milliseconds
    """
    start = time.perf_counter()
    artifact = load_artifact(artifact_path)
    elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
    return elapsed


def run_latency_benchmarks(artifact_path: str, query: str = "test query") -> dict:
    """
    Run all latency benchmarks.

    Args:
        artifact_path: Path to artifact directory
        query: Query string to use

    Returns:
        Dict with benchmark results
    """
    # Cold start
    cold_start_time = measure_cold_start_time(artifact_path)

    # Load artifact (warm)
    artifact = load_artifact(artifact_path)

    # Retrieval-only
    retrieval_latency = measure_retrieval_latency(artifact, query)

    # Retrieval+fetch
    retrieval_fetch_latency = measure_retrieval_fetch_latency(artifact, query)

    # Get artifact stats
    if artifact.manifest:
        total_chunks = artifact.manifest.total_chunks
        total_docs = artifact.manifest.total_docs
    else:
        total_chunks = 0
        total_docs = 0

    artifact_size = sum(
        f.stat().st_size
        for f in Path(artifact_path).iterdir()
        if f.is_file()
    ) / (1024 * 1024)  # MB

    return {
        "cold_start_ms": cold_start_time,
        "retrieval_only": retrieval_latency,
        "retrieval_fetch": retrieval_fetch_latency,
        "artifact_stats": {
            "total_chunks": total_chunks,
            "total_docs": total_docs,
            "artifact_size_mb": artifact_size,
        },
    }
