"""Benchmarks for Veritas RAG."""

from veritas_rag.benchmarks.latency import run_latency_benchmarks
from veritas_rag.benchmarks.portability import run_portability_benchmarks
from veritas_rag.benchmarks.quality import run_quality_benchmarks

__all__ = [
    "run_latency_benchmarks",
    "run_portability_benchmarks",
    "run_quality_benchmarks",
]
