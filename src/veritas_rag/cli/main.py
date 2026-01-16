"""
Main CLI entry point for Veritas RAG.
"""

import click
from typing import List

from veritas_rag import build_artifact, load_artifact
from veritas_rag.core import Config
from veritas_rag.core.contracts import RetrievalResult


@click.group()
def cli():
    """Veritas RAG - Local-first RAG memory engine."""
    pass


def normalize_scores_for_display(results: List[RetrievalResult]) -> List[float]:
    """
    Normalize BM25 scores for display (shift-to-zero).
    
    Preserves ranking order, guarantees non-negative values.
    Does NOT alter retrieval correctness - only affects display.
    """
    if not results:
        return []
    min_score = min(r.score for r in results)
    shift = -min_score if min_score < 0 else 0
    return [r.score + shift for r in results]


@cli.command()
@click.argument("corpus_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", required=True, type=click.Path())
@click.option("--chunk-size", default=512, type=int, help="Chunk size in words")
@click.option("--overlap", default=50, type=int, help="Chunk overlap in words")
def build(corpus_dir, output, chunk_size, overlap):
    """Build artifact from corpus."""
    config = Config(chunk_size=chunk_size, chunk_overlap=overlap)
    click.echo(f"Building artifact from {corpus_dir}...")
    build_artifact(corpus_dir, output, config)
    click.echo(f"Artifact built successfully at {output}")


@cli.command()
@click.argument("artifact_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("query", type=str)
@click.option("--top-k", default=10, type=int, help="Number of results")
@click.option("--format", "output_format", default="text", type=click.Choice(["text", "json"]))
@click.option("--strict", is_flag=True, help="Validate manifest checksums before querying")
@click.option("--show-raw-scores", is_flag=True, help="Show raw BM25 scores alongside normalized scores")
def query(artifact_dir, query, top_k, output_format, strict, show_raw_scores):
    """Query artifact."""
    artifact = load_artifact(artifact_dir, strict=strict)
    results = artifact.retrieve(query, top_k)

    if output_format == "json":
        import json

        output = [
            {
                "chunk_id": r.chunk_id,
                "score": r.score,  # Keep raw score in JSON
                "matched_terms": r.matched_terms,
                "snippet": r.snippet,
            }
            for r in results
        ]
        click.echo(json.dumps(output, indent=2))
    else:
        # Normalize scores for display
        display_scores = normalize_scores_for_display(results)
        
        for i, result in enumerate(results, 1):
            display_score = display_scores[i - 1]
            if show_raw_scores:
                click.echo(f"\n{i}. Score: {display_score:.4f} (raw: {result.score:.4f})")
            else:
                click.echo(f"\n{i}. Score: {display_score:.4f}")
            click.echo(f"   Chunk ID: {result.chunk_id}")
            click.echo(f"   Matched terms: {', '.join(result.matched_terms)}")


@cli.command()
@click.argument("artifact_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--suite", type=click.Choice(["latency", "quality", "portability", "all"]), default="all")
@click.option("--query", default="test query", help="Query string for latency benchmarks")
@click.option("--report-json", type=click.Path(), help="Write JSON report to file")
@click.option("--eval-set", type=click.Path(exists=True), default=None, help="Path to eval_set.jsonl (JSONL). If not set, use bundled default eval set.")
@click.option("--max-queries", type=int, default=None, help="Maximum number of queries to evaluate (samples if eval set is larger, for faster benchmarking)")
def benchmark(artifact_dir, suite, query, report_json, eval_set, max_queries):
    """Run benchmarks on artifact."""
    click.echo(f"Running benchmarks on {artifact_dir}...")

    if suite in ["latency", "all"]:
        from veritas_rag.benchmarks.latency import run_latency_benchmarks

        results = run_latency_benchmarks(artifact_dir, query, report_json_path=report_json)

        click.echo("\n=== Latency Benchmarks ===")
        click.echo(f"Cold start: {results['cold_start_ms']:.2f} ms")
        click.echo(f"\nRetrieval-only (P50/P95): {results['retrieval_only']['p50']:.2f} ms / {results['retrieval_only']['p95']:.2f} ms")
        click.echo(f"Retrieval+fetch (P50/P95): {results['retrieval_fetch']['p50']:.2f} ms / {results['retrieval_fetch']['p95']:.2f} ms")
        click.echo(f"\nArtifact stats:")
        artifact_stats = results.get('artifact', {})
        click.echo(f"  Total chunks: {artifact_stats.get('total_chunks', 'N/A')}")
        click.echo(f"  Total docs: {artifact_stats.get('total_docs', 'N/A')}")
        artifact_size = artifact_stats.get('size_mb', 0.0)
        click.echo(f"  Artifact size: {artifact_size:.2f} MB")

    if suite in ["portability", "all"]:
        from veritas_rag.benchmarks.portability import run_portability_benchmarks

        results = run_portability_benchmarks(artifact_dir, report_json_path=report_json)

        click.echo("\n=== Portability Benchmarks ===")
        click.echo(f"Cold start: {results['cold_start_ms']:.2f} ms")
        click.echo(f"\nArtifact sizes:")
        for file, size_mb in results["artifact_sizes_mb"].items():
            click.echo(f"  {file}: {size_mb:.2f} MB")
        click.echo(f"\nTotal chunks: {results['total_chunks']}")
        click.echo(f"Total docs: {results['total_docs']}")

    if suite in ["quality", "all"]:
        from veritas_rag.benchmarks.quality import run_quality_benchmarks

        results = run_quality_benchmarks(
            artifact_dir, 
            eval_set_path=eval_set, 
            report_json_path=report_json,
            max_queries=max_queries,
        )

        if "error" in results:
            click.echo(f"\n=== Quality Benchmarks ===")
            click.echo(f"Error: {results['error']}")
        else:
            click.echo("\n=== Quality Benchmarks ===")
            if results.get('sampled'):
                click.echo(f"Queries evaluated: {results['num_queries']} (sampled from {results.get('original_query_count', 'N/A')})")
            else:
                click.echo(f"Queries evaluated: {results['num_queries']}")
            click.echo(f"Recall@5: {results['recall_at_5']:.3f}")
            click.echo(f"Recall@10: {results['recall_at_10']:.3f}")
            click.echo(f"MRR: {results['mrr']:.3f}")


@cli.command("benchmark-gen")
@click.option("--output", "-o", required=True, type=click.Path())
@click.option("--target-chunks", default=10000, type=int, help="Target number of chunks")
@click.option("--num-docs", type=int, help="Number of documents (auto if not set)")
@click.option("--emit-eval", type=click.Path(), help="Write matching eval set to JSONL file")
def benchmark_gen(output, target_chunks, num_docs, emit_eval):
    """Generate synthetic corpus for benchmarking."""
    from pathlib import Path
    from veritas_rag.benchmarks.synth_corpus import generate_synthetic_corpus, generate_synthetic_eval_set

    corpus_path, corpus_meta = generate_synthetic_corpus(Path(output), target_chunks, num_docs)
    click.echo(f"Synthetic corpus generated at {corpus_path}")
    
    # Generate eval set if requested
    if emit_eval:
        corpus_meta_path = corpus_path / "corpus_meta.json"
        eval_path = generate_synthetic_eval_set(corpus_meta_path, Path(emit_eval))
        click.echo(f"Eval set generated at {eval_path}")


@cli.command("bench-run")
@click.option("--bucket", type=click.Choice(["10k", "100k"]), required=True)
@click.option("--report-json", type=click.Path(), required=True)
@click.option("--work-dir", type=click.Path(), default="/tmp", help="Working directory for temp files")
@click.option("--config", type=click.Path(), help="Config file (optional, uses defaults)")
@click.option("--max-queries", type=int, default=1000, help="Maximum queries for quality benchmark (default: 1000 for fast benchmarking)")
@click.option("--full-eval", is_flag=True, help="Use all queries in eval set (overrides --max-queries)")
def bench_run(bucket, report_json, work_dir, config, max_queries, full_eval):
    """Run standard benchmark bucket and generate combined report."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path
    
    from veritas_rag import build_artifact
    from veritas_rag.benchmarks.latency import run_latency_benchmarks
    from veritas_rag.benchmarks.portability import run_portability_benchmarks
    from veritas_rag.benchmarks.quality import run_quality_benchmarks
    from veritas_rag.benchmarks.reporting import collect_hardware_info
    from veritas_rag.benchmarks.synth_corpus import generate_synthetic_corpus, generate_synthetic_eval_set
    from veritas_rag.core import Config
    
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    
    # Determine target chunks
    target_chunks = 10000 if bucket == "10k" else 100000
    
    click.echo(f"Running {bucket} bucket benchmark...")
    
    # 1. Generate corpus with --emit-eval
    corpus_dir = work_path / f"corpus_{bucket}"
    eval_set_path = corpus_dir / "eval_set.jsonl"
    
    click.echo(f"Generating corpus with {target_chunks} target chunks...")
    corpus_path, corpus_meta = generate_synthetic_corpus(
        corpus_dir,
        target_chunks=target_chunks,
    )
    
    # Generate eval set (limit queries unless --full-eval)
    corpus_meta_path = corpus_path / "corpus_meta.json"
    eval_max_queries = None if full_eval else max_queries
    generate_synthetic_eval_set(corpus_meta_path, eval_set_path, max_queries=eval_max_queries)
    click.echo(f"Corpus and eval set generated at {corpus_path}")
    if eval_max_queries:
        click.echo(f"Eval set limited to {eval_max_queries} queries for faster benchmarking (use --full-eval for all queries)")
    
    # 2. Build artifact (capture config)
    artifact_dir = work_path / f"artifact_{bucket}"
    click.echo(f"Building artifact...")
    
    # Use provided config or defaults
    if config:
        # Load config from file (simplified - assume JSON)
        with open(config, "r") as f:
            config_dict = json.load(f)
        build_config = Config(**config_dict)
    else:
        build_config = Config()  # Use defaults
    
    build_artifact(str(corpus_path), str(artifact_dir), build_config)
    click.echo(f"Artifact built at {artifact_dir}")
    
    # 3. Run all benchmark suites
    click.echo("Running benchmarks...")
    
    errors = []
    
    # Latency
    try:
        latency_results = run_latency_benchmarks(str(artifact_dir), "test query", report_json_path=None)
    except Exception as e:
        errors.append({"benchmark": "latency", "error": str(e)})
        latency_results = {}
    
    # Portability
    try:
        portability_results = run_portability_benchmarks(str(artifact_dir), report_json_path=None)
    except Exception as e:
        errors.append({"benchmark": "portability", "error": str(e)})
        portability_results = {}
    
    # Quality (with synth eval)
    try:
        quality_results = run_quality_benchmarks(
            str(artifact_dir), 
            str(eval_set_path), 
            report_json_path=None,
            max_queries=None if full_eval else max_queries,
        )
    except Exception as e:
        errors.append({"benchmark": "quality", "error": str(e)})
        quality_results = {"error": str(e)}
    
    # 4. Write combined JSON report with generator + system params
    report_path = Path(report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract generator config from corpus_meta
    generator_config = corpus_meta.get("generator_config", {})
    
    # Extract system config from build_config
    system_config = {
        "chunk_size": build_config.chunk_size,
        "chunk_overlap": build_config.chunk_overlap,
        "bm25_k1": build_config.bm25_k1,
        "bm25_b": build_config.bm25_b,
    }
    
    # Combine all results
    combined_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "suite": "combined",
        "bucket": bucket,
        "hardware": collect_hardware_info(),
        "generator_config": generator_config,
        "system_config": system_config,
        "latency": {
            "cold_start_ms": latency_results.get("cold_start_ms", 0),
            "retrieval_only_p50": latency_results.get("retrieval_only", {}).get("p50", 0),
            "retrieval_only_p95": latency_results.get("retrieval_only", {}).get("p95", 0),
            "retrieval_fetch_p50": latency_results.get("retrieval_fetch", {}).get("p50", 0),
            "retrieval_fetch_p95": latency_results.get("retrieval_fetch", {}).get("p95", 0),
        },
        "portability": {
            "cold_start_ms": portability_results.get("cold_start_ms", 0),
            "artifact_sizes_mb": portability_results.get("artifact_sizes_mb", {}),
        },
        "quality": {
            "eval_set_path": str(eval_set_path),
            "recall_at_5": quality_results.get("recall_at_5", 0),
            "recall_at_10": quality_results.get("recall_at_10", 0),
            "recall_at_50": quality_results.get("recall_at_50", 0),
            "mrr": quality_results.get("mrr", 0),
            "category_breakdown": quality_results.get("category_breakdown", {}),
        },
        "artifact": {
            "total_chunks": latency_results.get("artifact", {}).get("total_chunks", 0) or portability_results.get("total_chunks", 0),
            "total_docs": latency_results.get("artifact", {}).get("total_docs", 0) or portability_results.get("total_docs", 0),
            "size_mb": latency_results.get("artifact", {}).get("size_mb", 0) or portability_results.get("artifact_stats", {}).get("size_mb", 0),
        },
    }
    
    # Add errors if any
    if errors:
        combined_report["errors"] = errors
    
    # Add multi-hop metrics if available
    if "recall_at_5_multi" in quality_results:
        combined_report["quality"]["recall_at_5_multi"] = quality_results["recall_at_5_multi"]
    if "recall_at_10_multi" in quality_results:
        combined_report["quality"]["recall_at_10_multi"] = quality_results["recall_at_10_multi"]
    
    # Add error from quality results if present
    if "error" in quality_results:
        combined_report["quality"]["error"] = quality_results["error"]
    
    # Write atomically to exact path
    temp_path = report_path.with_suffix('.tmp')
    try:
        with open(temp_path, "w") as f:
            json.dump(combined_report, f, indent=2)
        temp_path.rename(report_path)
        click.echo(f"\nWrote combined report to: {report_path}")
    except Exception as e:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise click.ClickException(f"Failed to write report to {report_path}: {e}")
    
    if errors:
        click.echo(f"Warning: {len(errors)} benchmark(s) failed. See 'errors' in report.")
    else:
        click.echo(f"Benchmark bucket {bucket} completed successfully!")


@cli.command("stat-card")
@click.argument("report_json", type=click.Path(exists=True))
def stat_card(report_json):
    """Generate printable stat card from benchmark report."""
    from pathlib import Path
    from veritas_rag.benchmarks.stat_card import generate_stat_card
    
    card = generate_stat_card(Path(report_json))
    click.echo(card)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
