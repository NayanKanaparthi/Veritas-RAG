# Fast Benchmarking Implementation

## Problem Solved

The original benchmark implementation was generating **~1.18 million queries** for a 10k chunk corpus, making quality benchmarks take 30+ minutes and timing out. This made it impractical to compare Veritas RAG with other frameworks like memvid.

## Solution Implemented

### 1. Query Sampling in Quality Benchmark
- Added `--max-queries` option to `benchmark` command
- Quality benchmark now samples queries proportionally by category if eval set is larger
- Preserves category distribution when sampling

### 2. Limited Eval Set Generation
- `generate_synthetic_eval_set()` now accepts `max_queries` parameter (default: 1000)
- Limits queries at generation time, not just evaluation time
- Reduces eval set file size from 273 MB to ~100 KB

### 3. Fast Default for bench-run
- `bench-run` now uses `--max-queries=1000` by default
- Completes in **~1-2 minutes** instead of 30+ minutes
- Add `--full-eval` flag to use all queries for comprehensive evaluation

## Usage

### Fast Benchmarking (Default)
```bash
# Fast benchmark with 1000 queries (~1-2 minutes)
veritas-rag bench-run --bucket 10k --report-json /tmp/report_10k.json
```

### Custom Query Limit
```bash
# Use 500 queries for even faster iteration
veritas-rag bench-run --bucket 10k --report-json /tmp/report_10k.json --max-queries 500
```

### Full Evaluation
```bash
# Use all queries in eval set (comprehensive but slow)
veritas-rag bench-run --bucket 10k --report-json /tmp/report_10k.json --full-eval
```

### Standalone Quality Benchmark
```bash
# Run quality benchmark with custom eval set and query limit
veritas-rag benchmark <artifact> --suite quality --eval-set <path> --max-queries 1000
```

## Results from Test Run

**Command**: `veritas-rag bench-run --bucket 10k --report-json /tmp/report_10k_fast.json`

**Completion Time**: ~1-2 minutes (vs 30+ minutes before)

**Results**:
- ✅ **Latency**: Excellent performance
  - Cold start: 236 ms
  - Retrieval-only P50: 2.57 ms
  - Retrieval+fetch P50: 4.16 ms
  
- ✅ **Portability**: Good compression
  - Total size: 64.26 MB for 12,632 chunks
  - Compression ratio: ~5 KB per chunk

- ⚠️ **Quality**: Very low scores (needs investigation)
  - Recall@10: 0.001 (1 out of 1000 queries)
  - This suggests source_path matching or offset resolution may need tuning
  - But benchmark completes successfully and quickly

## Comparison with Other Frameworks

Now you can:
1. Run fast benchmarks in 1-2 minutes
2. Compare latency/portability metrics directly
3. Iterate quickly on quality improvements
4. Use `--full-eval` for final comprehensive evaluation

## Files Modified

1. `src/veritas_rag/benchmarks/quality.py`
   - Added `max_queries` and `sample_seed` parameters
   - Implemented proportional category sampling
   - Added sampling metadata to results

2. `src/veritas_rag/benchmarks/synth_corpus.py`
   - Added `max_queries` parameter to `generate_synthetic_eval_set()`
   - Limits queries at generation time
   - Preserves category distribution

3. `src/veritas_rag/cli/main.py`
   - Added `--max-queries` option to `benchmark` command
   - Added `--max-queries` and `--full-eval` options to `bench-run` command
   - Updated quality benchmark output to show sampling info

## Next Steps for Quality Benchmark

The quality scores are very low (0.001 Recall@10), which suggests:
1. Source path matching may need refinement
2. Offset overlap logic may need adjustment
3. BM25 tokenization may not match synthetic corpus vocabulary well

But the **benchmarking infrastructure is now fast and ready** for iterative development and comparison with other frameworks.
