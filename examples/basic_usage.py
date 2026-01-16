"""
Basic usage example for Veritas RAG.
"""

from veritas_rag import build_artifact, load_artifact
from veritas_rag.core import Config

# Build artifact from corpus
print("Building artifact...")
config = Config(chunk_size=512, chunk_overlap=50)
build_artifact("corpus/", "artifact/", config)
print("Artifact built successfully!")

# Load artifact
print("\nLoading artifact...")
artifact = load_artifact("artifact/")

# Query
print("\nQuerying...")
results = artifact.retrieve("your query here", top_k=10)

for i, result in enumerate(results, 1):
    print(f"\n{i}. Score: {result.score:.4f}")
    print(f"   Chunk ID: {result.chunk_id}")
    print(f"   Matched terms: {', '.join(result.matched_terms)}")

# Fetch chunks
print("\nFetching chunks...")
chunk_ids = [r.chunk_id for r in results]
chunks = artifact.fetch_chunks(chunk_ids)

for chunk in chunks:
    print(f"\nChunk: {chunk.text[:100]}...")
    print(f"Source: {chunk.source_ref.source_path}")

# Generate answer (MVP stub)
print("\nGenerating answer...")
answer = artifact.answer("your query here", top_k=5)
print(answer)
