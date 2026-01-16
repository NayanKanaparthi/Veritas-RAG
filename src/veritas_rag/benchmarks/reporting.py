"""Shared utilities for benchmark reporting."""

import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import psutil
except ImportError:
    psutil = None


def collect_hardware_info() -> Dict[str, Any]:
    """Collect hardware and system information."""
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }

    # CPU info
    try:
        info["cpu"] = platform.processor() or platform.machine()
    except Exception:
        info["cpu"] = "unknown"

    # RAM info (if psutil available)
    if psutil:
        try:
            ram_gb = psutil.virtual_memory().total / (1024**3)
            info["ram_gb"] = round(ram_gb, 2)
        except Exception:
            info["ram_gb"] = None
    else:
        info["ram_gb"] = None

    return info


def write_json_report(
    report_path: Path,
    suite_name: str,
    metrics: Dict[str, Any],
    artifact_stats: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write benchmark JSON report.

    Args:
        report_path: Path to write JSON report
        suite_name: Name of benchmark suite (latency, portability, quality)
        metrics: Benchmark metrics
        artifact_stats: Artifact statistics (total_docs, total_chunks, size_mb)
        config: Configuration used (chunk_size, chunk_overlap, etc.)
    """
    hardware = collect_hardware_info()

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "suite": suite_name,
        "hardware": hardware,
        "metrics": metrics,
    }

    if artifact_stats:
        report["artifact"] = artifact_stats

    if config:
        report["config"] = config

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
