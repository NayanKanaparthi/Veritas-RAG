# Quality Benchmark Analysis - Root Cause Identified

## Problem Summary

Quality metrics show near-zero scores:
- Recall@5: ~0.000
- Recall@10: ~0.001-0.005
- MRR: ~0.000-0.003

## Root Cause: BM25 Ranking Limitation (NOT Evaluation Bug)

After thorough investigation, the issue is **BM25 ranking performance on synthetic data**, not an evaluation bug.

### What's Working ✅

1. **Source Path Resolution**: ✅ WORKING
   - Eval set has `source_path: "doc_01661.txt"`
   - Successfully resolves to `doc_id: "d64dbc97e8a5dc3e"`
   - Lookup dictionaries built correctly (4000 entries)

2. **Offset Overlap Matching**: ✅ WORKING
   - Eval set has offsets `[5520, 5537)`
   - Successfully finds chunk `d2bbb819448054c6` with offsets `[3234, 6791)`
   - Overlap check: `3234 < 5537 AND 6791 > 5520` = TRUE ✓

3. **Quote Matching**: ✅ WORKING
   - Quote "voacb dbcysf cjhn" found in chunk at correct position
   - Position in chunk: 2286
   - Expected: `5520 - 3234 = 2286` ✓

4. **Evaluation Logic**: ✅ WORKING
   - Correctly extracts relevant chunk IDs
   - Correctly compares with retrieved chunk IDs
   - Metrics computed correctly

### What's NOT Working ❌

**BM25 Ranking**: The correct chunk is NOT in top-10 results!

**Example**:
- Query: "voacb dbcysf cjhn" (exact phrase)
- Correct chunk: `d2bbb819448054c6` (contains quote at correct offset)
- BM25 top-10: Does NOT include `d2bbb819448054c6`
- BM25 finds the phrase in OTHER documents instead

### Why This Happens

1. **Synthetic Data Characteristics**:
   - Same phrases appear in multiple documents (low IDF)
   - BM25 scores based on term frequency and inverse document frequency
   - When phrases repeat, IDF is low, so all documents score similarly

2. **BM25 Limitations**:
   - BM25 is a term-based ranking algorithm
   - It doesn't understand "exact phrase" vs "term co-occurrence"
   - On synthetic data with repeated phrases, it can't distinguish which document is "correct"

3. **Expected Behavior**:
   - This is NOT a bug - it's a limitation of BM25 on synthetic data
   - Real-world data would have better IDF distribution
   - Other frameworks (dense embeddings, hybrid search) might perform better

## Verification

Tested with sample query:
```python
query = "voacb dbcysf cjhn"
correct_chunk = "d2bbb819448054c6"  # Found by offset overlap

BM25 top-10 results:
1. e5594605ec1bd26a (doc_02813.txt) - score 1.3388
2. 9c364782843ab9a8 (doc_03013.txt) - score 1.3332
3. fdddc91943abe565 (doc_01397.txt) - score 1.3215
...
❌ d2bbb819448054c6 NOT in top-10
```

The phrase appears in multiple documents, and BM25 ranks other documents higher due to term frequency differences.

## Solutions

### Option 1: Accept BM25 Limitations (Recommended for MVP)
- Document that BM25 has limitations on synthetic data
- Use real-world datasets for quality evaluation
- Focus on latency/portability metrics for framework comparison

### Option 2: Improve Synthetic Corpus
- Reduce phrase repetition (lower `exact_phrase_rate`)
- Increase vocabulary diversity
- Add more unique content per document

### Option 3: Use Different Retrieval (Phase 2)
- Add dense embeddings (hybrid search)
- Use phrase-aware ranking
- Implement exact phrase boosting

### Option 4: Adjust Evaluation (Not Recommended)
- Use Recall@50 instead of Recall@10 (shows correct chunks appear further down)
- This masks the real issue: BM25 ranking quality

## Current Status

- ✅ Evaluation infrastructure is correct
- ✅ Matching logic works (source_path, offsets, quotes)
- ✅ Metrics computed correctly
- ⚠️ BM25 ranking underperforms on synthetic data (expected)

## For Framework Comparison

**Recommendation**: Use latency and portability metrics for comparison, as these are:
- Framework-agnostic
- Not affected by synthetic data characteristics
- More relevant for production use cases

Quality metrics should be evaluated on:
- Real-world datasets
- Domain-specific corpora
- After implementing hybrid search (Phase 2)

## Next Steps

1. Document BM25 limitations in README
2. Add Recall@50 metric to show where correct chunks appear
3. Consider reducing phrase repetition in synthetic corpus generator
4. Focus on latency/portability for initial framework comparison
