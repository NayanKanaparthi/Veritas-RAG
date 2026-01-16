"""Stat card generator for benchmark reports."""

import json
from pathlib import Path
from typing import Dict


def generate_stat_card(report_json_path: Path) -> str:
    """
    Generate markdown-formatted stat card from benchmark report.
    
    Args:
        report_json_path: Path to combined JSON report
        
    Returns:
        Markdown-formatted stat card string
    """
    with open(report_json_path, "r") as f:
        report = json.load(f)
    
    lines = []
    lines.append("## Benchmark Results")
    lines.append("")
    
    # Hardware info
    hardware = report.get("hardware", {})
    if hardware:
        lines.append("### Hardware")
        lines.append(f"- CPU: {hardware.get('cpu', 'N/A')}")
        lines.append(f"- Memory: {hardware.get('memory_gb', 'N/A')} GB")
        lines.append(f"- Platform: {hardware.get('platform', 'N/A')}")
        lines.append("")
    
    # Corpus params
    generator_config = report.get("generator_config", {})
    if generator_config:
        lines.append("### Corpus Configuration")
        lines.append(f"- Vocab size: {generator_config.get('vocab_size', 'N/A')}")
        lines.append(f"- Entity injection rate: {generator_config.get('entity_injection_rate', 'N/A')}")
        lines.append(f"- Exact phrase rate: {generator_config.get('exact_phrase_rate', 'N/A')}")
        lines.append(f"- Distractor rate: {generator_config.get('distractor_rate', 'N/A')}")
        lines.append(f"- Seed: {generator_config.get('seed', 'N/A')}")
        lines.append(f"- Target chunks: {generator_config.get('target_chunks', 'N/A')}")
        lines.append("")
    
    # System params
    system_config = report.get("system_config", {})
    if system_config:
        lines.append("### System Configuration")
        lines.append(f"- Chunk size: {system_config.get('chunk_size', 'N/A')}")
        lines.append(f"- Chunk overlap: {system_config.get('chunk_overlap', 'N/A')}")
        lines.append(f"- BM25 k1: {system_config.get('bm25_k1', 'N/A')}")
        lines.append(f"- BM25 b: {system_config.get('bm25_b', 'N/A')}")
        lines.append("")
    
    # Artifact stats
    artifact = report.get("artifact", {})
    if artifact:
        lines.append("### Artifact Statistics")
        lines.append(f"- Total chunks: {artifact.get('total_chunks', 'N/A')}")
        lines.append(f"- Total docs: {artifact.get('total_docs', 'N/A')}")
        lines.append(f"- Artifact size: {artifact.get('size_mb', 'N/A'):.2f} MB")
        lines.append("")
    
    # Latency metrics
    latency = report.get("latency", {})
    if latency:
        lines.append("### Latency Metrics")
        lines.append(f"- Cold start: {latency.get('cold_start_ms', 'N/A'):.2f} ms")
        lines.append(f"- Retrieval-only (P50/P95): {latency.get('retrieval_only_p50', 'N/A'):.2f} ms / {latency.get('retrieval_only_p95', 'N/A'):.2f} ms")
        lines.append(f"- Retrieval+fetch (P50/P95): {latency.get('retrieval_fetch_p50', 'N/A'):.2f} ms / {latency.get('retrieval_fetch_p95', 'N/A'):.2f} ms")
        lines.append("")
    
    # Portability metrics
    portability = report.get("portability", {})
    if portability:
        lines.append("### Portability Metrics")
        lines.append(f"- Cold start: {portability.get('cold_start_ms', 'N/A'):.2f} ms")
        sizes = portability.get("artifact_sizes_mb", {})
        if sizes:
            lines.append("- Artifact file sizes:")
            for file, size in sizes.items():
                lines.append(f"  - {file}: {size:.2f} MB")
        lines.append("")
    
    # Quality metrics
    quality = report.get("quality", {})
    if quality:
        lines.append("### Quality Metrics")
        if "error" in quality:
            lines.append(f"- Error: {quality.get('error', 'N/A')}")
        else:
            recall_5 = quality.get('recall_at_5')
            recall_10 = quality.get('recall_at_10')
            mrr = quality.get('mrr')
            if recall_5 is not None:
                lines.append(f"- Recall@5: {recall_5:.3f}")
            else:
                lines.append("- Recall@5: N/A")
            if recall_10 is not None:
                lines.append(f"- Recall@10: {recall_10:.3f}")
            else:
                lines.append("- Recall@10: N/A")
            recall_50 = quality.get('recall_at_50')
            if recall_50 is not None:
                lines.append(f"- Recall@50: {recall_50:.3f}")
            if mrr is not None:
                lines.append(f"- MRR: {mrr:.3f}")
            else:
                lines.append("- MRR: N/A")
            
            # Show diagnostics if available
            diagnostics = quality.get('diagnostics')
            if diagnostics:
                lines.append("")
                lines.append("#### Diagnostics")
                lines.append(f"- Source path resolution failures: {diagnostics.get('resolution_failures', 0)}")
                lines.append(f"- Offset match failures: {diagnostics.get('offset_match_failures', 0)}")
                lines.append(f"- Quote match failures: {diagnostics.get('quote_match_failures', 0)}")
                lines.append(f"- Successful resolutions: {diagnostics.get('successful_resolutions', 0)}")
            
            # Multi-hop metrics
            if "recall_at_5_multi" in quality:
                lines.append(f"- Recall@5 (multi-hop): {quality.get('recall_at_5_multi'):.3f}")
            if "recall_at_10_multi" in quality:
                lines.append(f"- Recall@10 (multi-hop): {quality.get('recall_at_10_multi'):.3f}")
        
        # Category breakdown
        category_breakdown = quality.get("category_breakdown", {})
        if category_breakdown:
            lines.append("")
            lines.append("#### Category Breakdown")
            for category, metrics in category_breakdown.items():
                lines.append(f"- **{category}** (n={metrics.get('count', 0)}):")
                recall_5 = metrics.get('recall_at_5')
                recall_10 = metrics.get('recall_at_10')
                mrr = metrics.get('mrr')
                if recall_5 is not None:
                    lines.append(f"  - Recall@5: {recall_5:.3f}")
                if recall_10 is not None:
                    lines.append(f"  - Recall@10: {recall_10:.3f}")
                recall_50 = metrics.get('recall_at_50')
                if recall_50 is not None:
                    lines.append(f"  - Recall@50: {recall_50:.3f}")
                if mrr is not None:
                    lines.append(f"  - MRR: {mrr:.3f}")
        lines.append("")
    
    return "\n".join(lines)
