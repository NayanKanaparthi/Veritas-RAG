"""
Benchmark example for Veritas RAG.
"""

from veritas_rag import load_artifact

# Load artifact
artifact = load_artifact("artifact/")

# Simple latency benchmark
import time

query = "test query"
iterations = 100

# Warm-up
for _ in range(10):
    artifact.retrieve(query, top_k=10)

# Measure retrieval latency
times = []
for _ in range(iterations):
    start = time.time()
    results = artifact.retrieve(query, top_k=10)
    elapsed = (time.time() - start) * 1000  # Convert to ms
    times.append(elapsed)

# Calculate percentiles
times.sort()
p50 = times[len(times) // 2]
p95 = times[int(len(times) * 0.95)]

print(f"Retrieval latency (P50): {p50:.2f} ms")
print(f"Retrieval latency (P95): {p95:.2f} ms")
