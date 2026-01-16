# Veritas RAG

A local-first RAG memory engine that delivers fast, offline, portable, and trustworthy retrieval for LLM apps using sparse retrieval (BM25) plus a compact binary chunk store.

## Features

- **Sparse Retrieval**: BM25-based retrieval optimized for exact matches, proper nouns, and code symbols
- **Portable Artifacts**: Single directory artifact containing compressed chunks, index, and metadata
- **Deterministic IDs**: Content-based hashing ensures stable IDs across rebuilds
- **Offline-First**: No network required for querying
- **Integrity Checks**: Per-chunk and file-level checksums for data integrity
- **Atomic Builds**: Builds are atomic with validation

## Prerequisites

Before installing, ensure you have:
- **Python 3.10, 3.11, 3.12, or 3.13** (Python 3.14+ not yet supported)
- **pip** (Python package manager)
- **Git** (to clone the repository from GitHub)

## Installation

### Installation from Source (GitHub)

If you're installing from the GitHub repository:

```bash
# Clone the repository
git clone https://github.com/NayanKanaparthi/Veritas-RAG.git
cd Veritas-RAG

# Install in development mode (recommended)
python -m pip install -e ".[dev]"

# OR install in production mode (without dev dependencies)
python -m pip install -e .
```

This installs:
- The `veritas-rag` CLI command
- The Python API (`veritas_rag` package)
- All required dependencies (PyMuPDF, zstandard, rank-bm25, etc.)

### Installation from PyPI

Once published to PyPI:

```bash
pip install veritas-rag
```

### Development Install (for contributors)

If you're contributing to the project:

```bash
python -m pip install -e ".[dev]"
```

This installs the package in editable mode with development dependencies (pytest, mypy, ruff, etc.).

## Quick Start

### Using the CLI

```bash
# Build artifact from corpus
veritas-rag build corpus/ --output artifact/

# Query the artifact
veritas-rag query artifact/ "your query" --top-k 10

# Run benchmarks
veritas-rag benchmark artifact/ --suite all
```

### Using the Python API

```python
from veritas_rag import build_artifact, load_artifact
from veritas_rag.core import Config

# Build artifact from corpus
config = Config(chunk_size=512, chunk_overlap=50)
build_artifact("corpus/", "artifact/", config)

# Load and query
artifact = load_artifact("artifact/")
results = artifact.retrieve("your query", top_k=10)
chunks = artifact.fetch_chunks([r.chunk_id for r in results])
```

### Benchmark Data

The evaluation dataset (`eval_set_v1.jsonl`) is bundled with the package and automatically loaded via `importlib.resources` when running quality benchmarks. No manual path configuration needed.

## Getting Started

### Step-by-Step Workflow

#### 1. Prepare Your Corpus

Create a directory with your documents (PDFs and/or TXT files):

```bash
mkdir my_corpus
# Add your documents
cp document1.pdf my_corpus/
cp document2.txt my_corpus/
# ... etc
```

#### 2. Build an Artifact

```bash
veritas-rag build my_corpus/ --output my_artifact/
```

This will:
- Parse all PDFs and TXT files
- Normalize and chunk the text
- Build a BM25 search index
- Compress and store everything in `my_artifact/`

#### 3. Query the Artifact

```bash
# Simple query
veritas-rag query my_artifact/ "your search query" --top-k 10

# Query with JSON output
veritas-rag query my_artifact/ "your search query" --top-k 10 --format json

# Query with strict validation (checks checksums)
veritas-rag query my_artifact/ "your search query" --strict
```

## Common Use Cases

### Knowledge Base Search

Build a searchable knowledge base from documentation:

```bash
# Organize your documents
mkdir knowledge_base
cp docs/*.pdf knowledge_base/

# Build artifact
veritas-rag build knowledge_base/ --output kb_artifact/

# Query it
veritas-rag query kb_artifact/ "how do I configure X?" --top-k 5
```

### Code Documentation Search

Search through code documentation:

```bash
# Convert code docs to text files, then:
veritas-rag build code_docs/ --output docs_artifact/
veritas-rag query docs_artifact/ "function signature" --top-k 10
```

### Document Q&A System

Use Veritas RAG as the retrieval layer for a Q&A system:

```python
from veritas_rag import build_artifact, load_artifact

# One-time setup: build artifact
build_artifact("documents/", "artifact/", Config())

# Runtime: load and query
artifact = load_artifact("artifact/")

def answer_question(question: str):
    results = artifact.retrieve(question, top_k=5)
    chunks = artifact.fetch_chunks([r.chunk_id for r in results])
    
    # Combine chunks for LLM context
    context = "\n\n".join([chunk.text for chunk in chunks])
    
    # Pass to LLM (your implementation)
    # answer = llm.generate(context, question)
    return context
```

### Fast ID-Only Retrieval

For applications that only need chunk IDs (e.g., for caching):

```python
artifact = load_artifact("my_artifact/")

# Fast retrieval (no disk I/O, no decompression)
chunk_ids = artifact.retrieve_ids("query", top_k=10)

# Later, fetch chunks when needed
chunks = artifact.fetch_chunks(chunk_ids)
```

## Advanced Usage

### Custom Chunking Configuration

```bash
# Build with custom chunk size and overlap
veritas-rag build my_corpus/ --output my_artifact/ \
    --chunk-size 1024 \
    --overlap 100
```

Or via Python API:

```python
from veritas_rag.core import Config

config = Config(
    chunk_size=1024,      # Larger chunks
    chunk_overlap=100,    # More overlap
    bm25_k1=1.5,         # BM25 parameter
    bm25_b=0.75,         # BM25 parameter
    zstd_level=3         # Compression level
)
build_artifact("my_corpus/", "my_artifact/", config)
```

### Context Assembly

Get formatted context with citations:

```python
# Get formatted context with citations
context = artifact.context("your query", top_k=5)
print(context.text)  # Formatted text with citations
print(context.citations)  # List of source references
```

### Running Benchmarks

```bash
# Run all benchmarks
veritas-rag benchmark my_artifact/ --suite all

# Run only latency benchmarks
veritas-rag benchmark my_artifact/ --suite latency

# Run quality benchmarks with custom eval set
veritas-rag benchmark my_artifact/ --suite quality --eval-set my_eval_set.jsonl

# Generate synthetic corpus and run full benchmark suite
veritas-rag bench-run --bucket 10k --report-json results.json
```

## File Structure

### Artifact Directory

After building, your artifact directory will contain:

```
my_artifact/
├── manifest.json      # Artifact metadata and checksums
├── chunks.bin         # Compressed chunk data (zstd)
├── chunks.idx         # Binary chunk index
├── docs.meta          # Document metadata (JSON)
└── bm25_index.pkl     # BM25 search index (pickle)
```

**Note**: The entire artifact directory is portable and self-contained. You can share it, move it, or deploy it as-is.

### Corpus Directory

Your corpus directory should contain:
- **PDF files** (`.pdf`) - Automatically parsed
- **Text files** (`.txt`) - UTF-8 encoded

Subdirectories are supported and will be processed recursively.

## Troubleshooting

### Common Issues

#### "No module named 'veritas_rag'"

**Solution**: Make sure you installed the package:
```bash
cd veritas-rag
python -m pip install -e ".[dev]"
```

#### "Artifact not found" or "Invalid artifact"

**Solution**: Ensure the artifact directory contains all required files. Try rebuilding:
```bash
veritas-rag build corpus/ --output artifact/
```

#### "Pickle safety warning"

**This is normal**. The warning appears because `bm25_index.pkl` uses Python pickle format. Only load artifacts from trusted sources (pickle can execute arbitrary code). The warning is informational; the artifact will still load.

#### Low recall on queries

**BM25 works best for**:
- Exact phrase matches
- Proper nouns and entities
- Code symbols and technical terms

**Try**:
- Rephrasing queries to match document terminology
- Using exact phrases from your documents
- Increasing `--top-k` to retrieve more results

#### Build takes too long

**For large corpora**:
- The build process is single-threaded (Phase 2 will add parallelization)
- PDF parsing is the slowest step
- Consider preprocessing large PDFs or using smaller chunk sizes

### Getting Help

- Check the examples: See `examples/basic_usage.py` and `examples/benchmark_example.py`
- Review benchmarks: See `BENCHMARK_RESULTS.md` for performance characteristics
- Open an issue: Report bugs or ask questions on GitHub

## Architecture

- **Ingestion**: PDF/TXT parsing → normalization → chunking
- **Storage**: Compressed binary chunk store with manifest
- **Search**: BM25 sparse retrieval index
- **RAG**: Context assembly with citations (LLM integration in Phase 2)

## Artifact Contract

This section defines the core invariants and structure of Veritas RAG artifacts.

### Offsets

All `offset_start`/`offset_end` values refer to character positions in `Document.normalized_text`:
- **Inclusive start**: `offset_start` is the first character of the chunk
- **Exclusive end**: `offset_end` is one past the last character (Python slice convention)
- **Canonical reference**: Offsets always refer to `Document.normalized_text`, never to raw text

**Critical invariant**: `normalized_text[offset_start:offset_end] == chunk.text` for all chunks. This ensures stable chunk IDs and accurate citations.

### IDs

All IDs are deterministic content-based hashes:

- **`doc_uid`**: Stable hash of relative path (stable across content changes)
  - Format: `sha256(rel_path)[:16]`
  - Used for document identity across versions

- **`doc_id`**: Versioned hash of `doc_uid + normalized_text_hash`
  - Format: `sha256(doc_uid + normalized_text_sha256)[:16]`
  - Changes when document content changes

- **`chunk_id`**: Hash of `doc_uid + offset_start + offset_end + chunk_text_hash`
  - Format: `sha256(doc_uid + offset_start + offset_end + chunk_text_sha256)[:16]`
  - Stable as long as offsets and text remain unchanged

### Artifact Files

An artifact is a directory containing:

- **`chunks.bin`**: Compressed chunk payloads (append-only binary format, zstd compression)
- **`chunks.idx`**: Binary index with chunk metadata (chunk_id, doc_uid, doc_id, store_offset, length, checksum, is_active, offsets, page ranges)
- **`docs.meta`**: JSON document metadata (doc_uid, doc_id, source_path, title, extracted_at, normalized_text_sha256)
- **`bm25_index.pkl`**: Pickle-serialized BM25 index (MVP format; Phase 2 will use persistent inverted index)
- **`manifest.json`**: Artifact metadata and file-level SHA256 checksums

### Safety and Validation

**Pickle Safety Warning**: `bm25_index.pkl` uses Python pickle format. **Do not load artifacts from untrusted sources**; pickle can execute arbitrary code. The loader will display a warning by default.

**Validation Modes**:
- **Normal mode**: Basic schema version check (warns on mismatch)
- **Strict mode** (`--strict`): Validates all file checksums and chunk store invariants
- **Legacy mode** (`--allow-legacy`): Allows loading older schema versions (not recommended)

**Strict mode checks**:
- All required files exist
- SHA256 checksums match for all files
- Chunk store offsets don't exceed `chunks.bin` size
- Tombstoned chunks aren't returned by APIs

### Benchmark Definitions

When interpreting benchmark results, understand the metric categories:

- **Retrieval-only**: Query → top-k chunk IDs (no disk reads, no decompression)
  - Measures: BM25 tokenization, scoring, ranking
  - Use `artifact.retrieve_ids(query, top_k)` for this measurement

- **Retrieval+fetch**: Query → chunk text (includes disk reads and decompression)
  - Measures: Retrieval + `fetch_chunks()` (decompression, metadata loading)
  - Use `artifact.retrieve()` + `artifact.fetch_chunks()` for this measurement

- **End-to-end**: Query → final answer (includes LLM if enabled)
  - Measures: Retrieval + fetch + LLM generation + citation formatting
  - Use `artifact.answer(query, top_k)` for this measurement

## Status

MVP (Phase 1) in development. See plan for details.
