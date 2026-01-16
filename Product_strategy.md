# Veritas RAG

## 1) The business problem

Most RAG stacks assume you can push documents to a cloud vector database and keep an always-on retrieval service running.

That breaks in three common real-world situations:

- **Privacy & compliance**: contracts, internal docs, student data, enterprise IP cannot leave the device or controlled environment
- **Cost & operations**: hosted vector DB + ingestion pipelines + uptime monitoring add recurring cost and engineering overhead
- **Latency & reliability**: calling external services adds network variability, outages, and slow startup for user-facing experiences

## 2) The business outcome we want

Veritas RAG is built to deliver three outcomes that matter in product and strategy conversations:

### Outcome A — Lower total cost of ownership (TCO)

Ship a compact retrieval artifact instead of running retrieval infrastructure. That reduces:
- infra spend (DB hosting, scaling, networking)
- engineering time (ops, monitoring, debugging distributed systems)
- deployment complexity (fewer dependencies, fewer failure modes)

### Outcome B — Privacy and "data stays local" trust

Knowledge can remain on-device or in a controlled environment. This enables:
- enterprise adoption where data export is restricted
- offline/edge use cases
- clear privacy messaging (a product differentiator)

### Outcome C — Predictable, fast user experience

Local retrieval avoids network roundtrips and gives stable performance:
- low-ms retrieval enables interactive UX
- fast startup supports CLI tools and lightweight apps
- deterministic behavior improves reliability

## 3) Solution

Veritas RAG is a portable, local-first retrieval system that packages a corpus into a single "retrieval artifact" you can ship to any machine and query with low latency, without running a vector database, external services, or network calls.

**Think: "sqlite for retrieval."** A compact bundle containing chunked text, document metadata, a lexical index (BM25 in v1), and fast lookup structures for fetch.

### Why I built it (the strategist vision)

Most RAG stacks are optimized for cloud scale (hosted vector DBs, managed pipelines). But many real-world constraints optimize for privacy, cost, portability, and deterministic operations:

- **Privacy / compliance**: sensitive docs cannot leave device or controlled environment
- **Cost & simplicity**: no always-on DB or managed service bills
- **Portability**: ship knowledge with the app; offline-ready
- **Determinism**: predictable latency, fewer moving parts, reproducibility

**Goal**: make retrieval an artifact you build once and query anywhere with stable performance.

### Design goals & constraints

#### Primary goals

- Fast local retrieval (single-digit ms top-k search)
- Portable artifact (copy a folder; no server dependency)
- Low operational complexity (no external DB)
- Transparent benchmarking (latency + quality + artifact size)

#### Non-goals in v1

- Best-in-class semantic retrieval quality (BM25-only has known limits)
- Agentic multi-hop reasoning or heavy reranking pipelines

### High-level architecture

**Build time (offline)**: normalize docs → chunk with offsets → write chunk store + doc metadata → build BM25 index → write manifest

**Query time (online)**: tokenize query → BM25 search returns top-k chunk IDs → fetch chunk text + metadata from chunk store → return results

### Artifact contents (portable folder)

- `chunks.bin`: contiguous chunk text storage
- `chunks.idx`: offsets into chunks.bin for O(1) chunk fetch
- `docs.meta`: mapping doc_uid → doc_id, source_path, etc.
- `bm25_index`: BM25 index (token stats + postings)
- `manifest`: build metadata and config

### Retrieval approach (v1)

#### Why BM25 first

BM25 is fast, interpretable, and has zero embedding cost. It is strong on exact phrase / keyword queries and is a good baseline to validate the artifact abstraction and benchmarking pipeline before adding semantic layers.

#### What happens at query time

1. Tokenize query
2. Score chunks via BM25
3. Return top-k chunk IDs with scores
4. Fetch text + metadata via chunk store

## Benchmarking philosophy (what "working" means)

I benchmark three dimensions because "RAG quality" alone is incomplete:

### Latency

- **Retrieval-only**: index search time
- **Retrieval+fetch**: end-to-end time to return chunk text

### Portability / Artifact size

- Total on-disk footprint and breakdown by file

### Quality

- Recall@k and MRR using synthetic eval sets with ground-truth relevance

### What the metrics mean (engineer-level clarity)

- **Recall@k**: For each query, did we retrieve at least one correct chunk in the top k results?
  - Example: Recall@10 = 0.16 means 16% of queries found a relevant chunk in the top 10.
- **MRR (Mean Reciprocal Rank)**: How high is the first correct result ranked?
  - If the first correct result is rank r, the score is 1/r. Averaged across queries.
- **Cold start**: time to load artifact + initialize index. Important for CLI tools and apps that need fast startup.

### Current results snapshot (10k bucket, local runs)

Representative run ranges (exact values vary with seed and corpus generation):

- **Artifact size**: ~65–73 MB for ~10k target chunks
- **Cold start**: ~230–260 ms
- **Retrieval-only P50**: ~2–3 ms
- **Retrieval+fetch P50**: ~3–4.5 ms

### Quality outcome (important)

When "exact phrase" injections include unique tokens (designed to be unambiguous), exact-phrase retrieval is strong. In that category, Recall@10 can hit 1.0 because BM25 reliably matches rare/unique tokens.

For paraphrase and entity-heavy queries, BM25-only collapses. These categories often show near-zero Recall@10 because lexical matching cannot capture semantic equivalence or entity resolution without additional signals.

**Interpretation**: the system is architecturally working (artifact build → load → retrieve → fetch is correct). The limitation is modeling, not infrastructure: BM25 can't solve semantic paraphrase/entity resolution on its own.

## Key trade-offs (what I chose and why)

### Trade-off 1: Portability vs best-possible quality

- **Chose**: portable local artifact + BM25 baseline
- **Gave up**: semantic recall in v1
- **Reason**: validate the "artifact" abstraction first; add semantic retrieval next without breaking portability

### Trade-off 2: Low ops vs managed services

- **Chose**: no vector DB, no server, no hosted infra
- **Benefit**: cheaper, simpler, privacy-friendly, offline-ready
- **Cost**: fewer out-of-the-box features (hybrid search, reranking, observability)

### Trade-off 3: Determinism vs adaptive pipelines

- **Chose**: deterministic indexing and retrieval
- **Benefit**: predictable latency and reproducibility
- **Cost**: less flexible than learning-based retrieval unless we add additional layers

## Roadmap (how this becomes advanced RAG)

To win on paraphrase/entity-heavy retrieval while preserving the artifact idea:

### Hybrid retrieval (BM25 + dense vectors)

- Add an embedding index (FAISS/HNSW) into the artifact
- Query both BM25 and dense index, merge candidates (weighted union)

### Lightweight reranking

- Cross-encoder reranker (local small model) or optional LLM rerank
- Rerank only top ~50 candidates to preserve latency

### Entity-aware normalization

- Better tokenization and optional entity dictionary
- Boost entity matches without losing semantic recall

### Hardening and safety

- Replace pickle-based persistence with safer formats or signed artifacts
- Versioned manifest + backward-compatible readers

## Business value (how I pitch it)

If you're building product features on top of retrieval, Veritas creates a foundation optimized for product constraints:

- **Cost**: no always-on vector DB, predictable storage footprint
- **Privacy**: keep knowledge on-device or inside a controlled environment
- **Speed**: low-ms retrieval enables interactive UX
- **Productization**: ship "knowledge packs" as artifacts (offline docs, client installs, on-device apps)

## Bottom line

Veritas is a systems-first RAG foundation: portable, fast, and low-ops. It intentionally starts with a strong lexical baseline to validate the artifact design and performance, and it is designed to evolve into hybrid semantic retrieval while preserving the core "retrieval-as-an-artifact" abstraction.
