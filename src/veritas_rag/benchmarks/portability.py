"""Portability benchmarks for Veritas RAG.

Measures:
- Cold start load time
- Artifact size
- Basic offline operation check
"""

import time
from pathlib import Path
from typing import Dict, Optional

from veritas_rag import load_artifact
from veritas_rag.benchmarks.reporting import write_json_report


def measure_cold_start_time(artifact_path: str) -> float:
    """Measure cold start time (load artifact from disk)."""
    start = time.perf_counter()
    artifact = load_artifact(artifact_path)
    elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
    return elapsed


def measure_artifact_size(artifact_path: str) -> Dict[str, float]:
    """Measure artifact size breakdown."""
    artifact_dir = Path(artifact_path)

    sizes = {}
    total = 0

    for file in ["chunks.bin", "chunks.idx", "bm25_index.pkl", "docs.meta", "manifest.json"]:
        file_path = artifact_dir / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            sizes[file] = size_mb
            total += size_mb

    sizes["total"] = total
    return sizes


def run_portability_benchmarks(artifact_path: str, report_json_path: Optional[str] = None) -> Dict:
    """Run all portability benchmarks."""
    cold_start = measure_cold_start_time(artifact_path)
    sizes = measure_artifact_size(artifact_path)

    # Load artifact to verify offline operation
    artifact = load_artifact(artifact_path)
    if artifact.manifest:
        total_chunks = artifact.manifest.total_chunks
        total_docs = artifact.manifest.total_docs
    else:
        total_chunks = 0
        total_docs = 0

    artifact_stats = {
        "total_chunks": total_chunks,
        "total_docs": total_docs,
        "size_mb": sizes.get("total", 0.0),
    }

    result = {
        "cold_start_ms": cold_start,
        "artifact_sizes_mb": sizes,
        "total_chunks": total_chunks,
        "total_docs": total_docs,
    }

    # Write JSON report if requested
    if report_json_path:
        metrics = {
            "cold_start_ms": cold_start,
        }
        write_json_report(
            Path(report_json_path),
            "portability",
            metrics,
            artifact_stats=artifact_stats,
        )

    return result
