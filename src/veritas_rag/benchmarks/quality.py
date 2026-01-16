"""Quality benchmarks for Veritas RAG.

Measures:
- Recall@k
- Recall@k_multi (for multi-hop queries)
- MRR (Mean Reciprocal Rank)
- Category breakdown
"""

import json
import warnings
from collections import defaultdict
from importlib import resources
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from veritas_rag import load_artifact
from veritas_rag.benchmarks.reporting import write_json_report


def load_eval_set(eval_set_path: str) -> List[Dict]:
    """
    Load evaluation set from JSONL file.
    
    Args:
        eval_set_path: Path to eval_set.jsonl file
        
    Returns:
        List of query dictionaries
        
    Raises:
        ValueError: If file is not JSONL format (.jsonl extension)
    """
    eval_path = Path(eval_set_path)
    
    # Check file extension
    if eval_path.suffix.lower() == ".json":
        raise ValueError("Eval set must be JSONL format (.jsonl), not JSON")
    
    queries = []
    try:
        with open(eval_set_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    queries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} of {eval_set_path}: {e}"
                    ) from e
    except FileNotFoundError:
        raise FileNotFoundError(f"Eval set file not found: {eval_set_path}")
    
    return queries


def compute_recall_at_k(relevant_ids: Set[str], retrieved_ids: List[str], k: int) -> float:
    """Compute Recall@k."""
    if not relevant_ids:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    return len(retrieved_k & relevant_ids) / len(relevant_ids)


def compute_recall_at_k_multi(
    relevant_items_list: List[Set[str]], retrieved_ids: List[str], k: int
) -> float:
    """
    Compute Recall@k for multi-hop queries.
    
    Query succeeds only if ALL relevant_items appear in top-k.
    """
    if not relevant_items_list:
        return 0.0
    
    retrieved_k = set(retrieved_ids[:k])
    
    # All relevant item sets must have at least one match in top-k
    for relevant_ids in relevant_items_list:
        if not (retrieved_k & relevant_ids):
            return 0.0
    
    return 1.0


def compute_mrr(relevant_ids: Set[str], retrieved_ids: List[str]) -> float:
    """Compute Mean Reciprocal Rank."""
    if not relevant_ids:
        return 0.0
    relevant_set = relevant_ids
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    return 0.0


def build_source_path_lookups(docs_meta: Dict) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Build multiple lookup dictionaries for source_path → doc_id resolution.
    
    Args:
        docs_meta: Document metadata from artifact
        
    Returns:
        Tuple of (by_exact, by_norm, by_name) dictionaries
    """
    by_exact = {}
    by_norm = {}
    by_name = {}
    
    for doc_info in docs_meta.values():
        source_path = doc_info.get("source_path")
        doc_id = doc_info.get("doc_id")
        
        if not source_path or not doc_id:
            continue
        
        # Exact match
        by_exact[source_path] = doc_id
        
        # Normalized posix path
        try:
            norm_path = Path(source_path).as_posix()
            by_norm[norm_path] = doc_id
        except Exception:
            pass
        
        # Basename match
        try:
            basename = Path(source_path).name
            by_name[basename] = doc_id
        except Exception:
            pass
    
    return by_exact, by_norm, by_name


def resolve_doc_id_from_source_path(
    source_path: str, by_exact: Dict[str, str], by_norm: Dict[str, str], by_name: Dict[str, str]
) -> Optional[str]:
    """
    Resolve doc_id from source_path using multiple fallback strategies.
    
    Args:
        source_path: Source path from eval set
        by_exact: Exact string match dictionary
        by_norm: Normalized posix path match dictionary
        by_name: Basename match dictionary
        
    Returns:
        doc_id if found, None otherwise
    """
    # Try exact match first
    if source_path in by_exact:
        return by_exact[source_path]
    
    # Try normalized posix path
    try:
        norm_path = Path(source_path).as_posix()
        if norm_path in by_norm:
            return by_norm[norm_path]
    except Exception:
        pass
    
    # Try basename match
    try:
        basename = Path(source_path).name
        if basename in by_name:
            return by_name[basename]
    except Exception:
        pass
    
    return None


def find_chunks_by_offset_overlap(
    artifact, doc_id: str, offset_start: int, offset_end: int
) -> List[str]:
    """
    Find chunk IDs that overlap with the given offset range.
    
    Overlap condition: chunk.offset_start < item.offset_end AND chunk.offset_end > item.offset_start
    """
    chunk_ids = []
    
    # Find doc_uid from doc_id
    docs_meta = artifact.chunk_store.load_docs_meta()
    doc_uid = None
    for uid, doc_info in docs_meta.items():
        if doc_info.get("doc_id") == doc_id:
            doc_uid = uid
            break
    
    if doc_uid is None:
        return chunk_ids
    
    # Get all chunks for this doc_uid
    if doc_uid not in artifact.chunk_store.doc_uid_to_chunks:
        return chunk_ids
    
    # Check each chunk for offset overlap
    for chunk_id in artifact.chunk_store.doc_uid_to_chunks[doc_uid]:
        chunk_record = artifact.chunk_store.index.get(chunk_id)
        if not chunk_record or not chunk_record.get("is_active", True):
            continue
        
        chunk_offset_start = chunk_record.get("offset_start")
        chunk_offset_end = chunk_record.get("offset_end")
        
        if chunk_offset_start is None or chunk_offset_end is None:
            continue
        
        # Check overlap: chunk.offset_start < item.offset_end AND chunk.offset_end > item.offset_start
        if chunk_offset_start < offset_end and chunk_offset_end > offset_start:
            chunk_ids.append(chunk_id)
    
    return chunk_ids


def find_chunks_by_quote(artifact, doc_id: str, quote: str) -> List[str]:
    """
    Find chunk IDs containing the quote text (fallback method).
    
    Uses exact match first, then case-insensitive if needed.
    """
    chunk_ids = []
    
    # Find doc_uid from doc_id
    docs_meta = artifact.chunk_store.load_docs_meta()
    doc_uid = None
    for uid, doc_info in docs_meta.items():
        if doc_info.get("doc_id") == doc_id:
            doc_uid = uid
            break
    
    if doc_uid is None:
        return chunk_ids
    
    # Get all chunks for this doc_uid
    if doc_uid not in artifact.chunk_store.doc_uid_to_chunks:
        return chunk_ids
    
    # Normalize quote (same as ingestion pipeline)
    from veritas_rag.ingestion.normalizer import normalize_text
    quote_normalized = normalize_text(quote)
    
    # Check each chunk for quote match
    for chunk_id in artifact.chunk_store.doc_uid_to_chunks[doc_uid]:
        try:
            chunk = artifact.chunk_store.read_chunk(chunk_id)
            if chunk:
                # Try exact match first
                if quote in chunk.text or quote_normalized in chunk.text:
                    chunk_ids.append(chunk_id)
                # Try case-insensitive
                elif quote.lower() in chunk.text.lower():
                    chunk_ids.append(chunk_id)
        except Exception:
            continue
    
    return chunk_ids


def run_quality_benchmarks(
    artifact_path: str, 
    eval_set_path: str = None, 
    report_json_path: Optional[str] = None,
    max_queries: Optional[int] = None,
    sample_seed: Optional[int] = None,
) -> Dict:
    """
    Run quality benchmarks using eval set with offset-overlap matching.
    
    Args:
        artifact_path: Path to artifact directory
        eval_set_path: Path to eval set JSONL file (uses default if None)
        report_json_path: Optional path to write JSON report
        max_queries: Maximum number of queries to evaluate (samples if eval set is larger)
        sample_seed: Random seed for query sampling (for reproducibility)
    """
    artifact = load_artifact(artifact_path)

    # Default eval set path - use importlib.resources for bundled data
    if eval_set_path is None:
        try:
            data_path = resources.files("veritas_rag.benchmarks.data").joinpath("eval_set_v1.jsonl")
            with resources.as_file(data_path) as p:
                eval_set_path = str(p)
        except (ModuleNotFoundError, FileNotFoundError):
            return {
                "error": "Eval set not found (not bundled or package not installed)",
                "eval_set_path": "veritas_rag.benchmarks.data/eval_set_v1.jsonl",
            }

    if not Path(eval_set_path).exists():
        return {
            "error": "Eval set not found",
            "eval_set_path": str(eval_set_path),
        }

    # Load and validate eval set
    try:
        queries = load_eval_set(eval_set_path)
    except ValueError as e:
        return {
            "error": str(e),
            "eval_set_path": str(eval_set_path),
        }
    except FileNotFoundError as e:
        return {
            "error": str(e),
            "eval_set_path": str(eval_set_path),
        }
    
    # Sample queries if max_queries specified
    original_query_count = len(queries)
    if max_queries and len(queries) > max_queries:
        import random
        if sample_seed is not None:
            random.seed(sample_seed)
        # Preserve category distribution when sampling
        queries_by_category = defaultdict(list)
        for q in queries:
            category = q.get("category", "unknown")
            queries_by_category[category].append(q)
        
        # Sample proportionally from each category
        sampled = []
        for category, cat_queries in queries_by_category.items():
            cat_max = max(1, int(max_queries * len(cat_queries) / original_query_count))
            sampled.extend(random.sample(cat_queries, min(cat_max, len(cat_queries))))
        
        # If we still have too many, randomly sample to reach exact max
        if len(sampled) > max_queries:
            sampled = random.sample(sampled, max_queries)
        
        queries = sampled
    
    # Load docs.meta and build robust source_path → doc_id lookups
    docs_meta = artifact.chunk_store.load_docs_meta()
    by_exact, by_norm, by_name = build_source_path_lookups(docs_meta)

    recall_at_5 = []
    recall_at_10 = []
    recall_at_50 = []  # Add Recall@50 to see if correct chunks appear further down
    recall_at_5_multi = []
    recall_at_10_multi = []
    mrr_scores = []
    skipped_count = 0
    
    # Diagnostic counters
    resolution_failures = 0
    offset_match_failures = 0
    quote_match_failures = 0
    successful_resolutions = 0
    
    # Category breakdown
    category_metrics = defaultdict(lambda: {
        "recall_at_5": [],
        "recall_at_10": [],
        "recall_at_50": [],
        "mrr": [],
        "count": 0,
    })

    for query_data in queries:
        query_text = query_data.get("query_text", "")
        relevant_items = query_data.get("relevant_items", [])
        category = query_data.get("category", "unknown")

        # Extract relevant chunk IDs
        relevant_chunk_ids_set = set()
        relevant_items_list = []  # For multi-hop: list of sets
        query_valid = True

        for item in relevant_items:
            item_chunk_ids = set()
            
            # Method 1: Use chunk_id if provided
            if "chunk_id" in item:
                chunk_id = item["chunk_id"]
                try:
                    chunk = artifact.chunk_store.read_chunk(chunk_id)
                    if chunk is None:
                        query_valid = False
                        break
                    item_chunk_ids.add(chunk_id)
                except Exception:
                    query_valid = False
                    break
            
            # Method 2: Use source_path + offset overlap
            elif "source_path" in item:
                source_path = item["source_path"]
                
                # Map source_path to doc_id using robust resolution
                doc_id = resolve_doc_id_from_source_path(source_path, by_exact, by_norm, by_name)
                
                # Fallback: try reverse lookup from doc_id_to_source_path
                if doc_id is None:
                    for d_id, sp in artifact.chunk_store.doc_id_to_source_path.items():
                        # Try exact match
                        if sp == source_path:
                            doc_id = d_id
                            break
                        # Try basename match
                        try:
                            if Path(sp).name == Path(source_path).name:
                                doc_id = d_id
                                break
                        except Exception:
                            pass
                
                if doc_id is None:
                    resolution_failures += 1
                    query_valid = False
                    break
                
                successful_resolutions += 1
                
                # Use offset overlap if available
                if "offset_start" in item and "offset_end" in item:
                    offset_start = item["offset_start"]
                    offset_end = item["offset_end"]
                    found_chunks = find_chunks_by_offset_overlap(
                        artifact, doc_id, offset_start, offset_end
                    )
                    item_chunk_ids.update(found_chunks)
                    
                    if not found_chunks:
                        offset_match_failures += 1
                
                # Fallback to quote matching (always try, even if offset match succeeded)
                # This ensures we find chunks even if offset calculation is slightly off
                if "relevant_quote" in item:
                    quote = item["relevant_quote"]
                    found_chunks = find_chunks_by_quote(artifact, doc_id, quote)
                    item_chunk_ids.update(found_chunks)
                    
                    if not found_chunks:
                        quote_match_failures += 1
            
            # Method 3: Use doc_id directly (legacy)
            elif "doc_id" in item:
                doc_id = item["doc_id"]
                if doc_id not in artifact.chunk_store.doc_id_to_source_path:
                    query_valid = False
                    break
                # For doc_id without offsets, we can't determine specific chunks
                # This is a legacy case - skip for now
                query_valid = False
                break
            
            if not item_chunk_ids:
                query_valid = False
                break
            
            relevant_chunk_ids_set.update(item_chunk_ids)
            relevant_items_list.append(item_chunk_ids)

        # Skip invalid queries
        if not query_valid or not relevant_chunk_ids_set:
            skipped_count += 1
            warnings.warn(
                f"Skipping query '{query_text[:50]}...' - references missing chunk_ids or doc_ids"
            )
            continue

        # Retrieve with larger top_k to check if correct chunks appear further down
        # This helps diagnose BM25 ranking issues on synthetic data
        results = artifact.retrieve_ids(query_text, top_k=50)  # Retrieve more for better recall
        retrieved_ids = [chunk_id for chunk_id, score in results]

        # Compute metrics (using top-10 for standard metrics, top-50 for diagnosis)
        recall_5 = compute_recall_at_k(relevant_chunk_ids_set, retrieved_ids, 5)
        recall_10 = compute_recall_at_k(relevant_chunk_ids_set, retrieved_ids, 10)
        recall_50 = compute_recall_at_k(relevant_chunk_ids_set, retrieved_ids, 50)
        mrr = compute_mrr(relevant_chunk_ids_set, retrieved_ids)
        
        recall_at_5.append(recall_5)
        recall_at_10.append(recall_10)
        recall_at_50.append(recall_50)
        mrr_scores.append(mrr)
        
        # Category breakdown
        category_metrics[category]["recall_at_5"].append(recall_5)
        category_metrics[category]["recall_at_10"].append(recall_10)
        category_metrics[category]["recall_at_50"].append(recall_50)
        category_metrics[category]["mrr"].append(mrr)
        category_metrics[category]["count"] += 1
        
        # Multi-hop scoring (if multi-hop query)
        if category == "multi-hop" and len(relevant_items_list) >= 2:
            recall_5_multi = compute_recall_at_k_multi(relevant_items_list, retrieved_ids, 5)
            recall_10_multi = compute_recall_at_k_multi(relevant_items_list, retrieved_ids, 10)
            recall_at_5_multi.append(recall_5_multi)
            recall_at_10_multi.append(recall_10_multi)

    # Check if too many queries were skipped
    skipped_ratio = skipped_count / len(queries) if queries else 0.0
    if skipped_ratio > 0.5:
        return {
            "error": f"Too many queries skipped ({skipped_ratio:.1%}). Eval set may not match artifact.",
            "num_queries": len(queries),
            "skipped": skipped_count,
            "skipped_ratio": skipped_ratio,
            "diagnostics": {
                "resolution_failures": resolution_failures,
                "offset_match_failures": offset_match_failures,
                "quote_match_failures": quote_match_failures,
                "successful_resolutions": successful_resolutions,
            },
        }

    # Compute category averages
    category_breakdown = {}
    for cat, metrics in category_metrics.items():
        if metrics["count"] > 0:
            category_breakdown[cat] = {
                "count": metrics["count"],
                "recall_at_5": sum(metrics["recall_at_5"]) / len(metrics["recall_at_5"]) if metrics["recall_at_5"] else 0.0,
                "recall_at_10": sum(metrics["recall_at_10"]) / len(metrics["recall_at_10"]) if metrics["recall_at_10"] else 0.0,
                "recall_at_50": sum(metrics["recall_at_50"]) / len(metrics["recall_at_50"]) if metrics["recall_at_50"] else 0.0,
                "mrr": sum(metrics["mrr"]) / len(metrics["mrr"]) if metrics["mrr"] else 0.0,
            }

    result = {
        "num_queries": len(queries),
        "original_query_count": original_query_count if max_queries else len(queries),
        "sampled": max_queries is not None and original_query_count > max_queries,
        "skipped": skipped_count,
        "recall_at_5": sum(recall_at_5) / len(recall_at_5) if recall_at_5 else 0.0,
        "recall_at_10": sum(recall_at_10) / len(recall_at_10) if recall_at_10 else 0.0,
        "recall_at_50": sum(recall_at_50) / len(recall_at_50) if recall_at_50 else 0.0,
        "mrr": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
        "category_breakdown": category_breakdown,
        "diagnostics": {
            "resolution_failures": resolution_failures,
            "offset_match_failures": offset_match_failures,
            "quote_match_failures": quote_match_failures,
            "successful_resolutions": successful_resolutions,
        },
    }
    
    # Add multi-hop metrics if available
    if recall_at_5_multi:
        result["recall_at_5_multi"] = sum(recall_at_5_multi) / len(recall_at_5_multi)
    if recall_at_10_multi:
        result["recall_at_10_multi"] = sum(recall_at_10_multi) / len(recall_at_10_multi)

    # Write JSON report if requested
    if report_json_path and "error" not in result:
        artifact_stats = None
        if artifact.manifest:
            artifact_stats = {
                "total_chunks": artifact.manifest.total_chunks,
                "total_docs": artifact.manifest.total_docs,
            }
        metrics = {
            "recall_at_5": result["recall_at_5"],
            "recall_at_10": result["recall_at_10"],
            "recall_at_50": result["recall_at_50"],
            "mrr": result["mrr"],
            "num_queries": result["num_queries"],
            "skipped": result.get("skipped", 0),
        }
        if "recall_at_5_multi" in result:
            metrics["recall_at_5_multi"] = result["recall_at_5_multi"]
        if "recall_at_10_multi" in result:
            metrics["recall_at_10_multi"] = result["recall_at_10_multi"]
        write_json_report(
            Path(report_json_path),
            "quality",
            metrics,
            artifact_stats=artifact_stats,
        )

    return result
