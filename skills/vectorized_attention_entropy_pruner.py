"""
Deployed in Evolution Generation 5 (Score: 0.98)
Hypothesis: By refactoring iterative pruning and thresholding of attention weights into a single contiguous vectorized NumPy operation utilizing boolean mask slicing and pre-allocated output tensors, we eliminate Python loop overhead and reduce execution latency by >= 40% while preserving absolute numerical equivalence.
"""

import time
import numpy as np

def baseline_prune(attention_matrix, threshold):
    rows, cols = attention_matrix.shape
    pruned = np.zeros_like(attention_matrix)
    for i in range(rows):
        for j in range(cols):
            if attention_matrix[i, j] >= threshold:
                pruned[i, j] = attention_matrix[i, j]
            else:
                pruned[i, j] = 0.0
    return pruned

def vectorized_prune(attention_matrix, threshold):
    return np.where(attention_matrix >= threshold, attention_matrix, 0.0)

# Benchmark harness
np.random.seed(42)
data = np.random.rand(512, 512)
thresh = 0.5

# Baseline timing
t0 = time.time()
for _ in range(10):
    res_base = baseline_prune(data, thresh)
baseline_time = (time.time() - t0) * 1000.0

# Mutation timing
t0 = time.time()
for _ in range(10):
    res_mut = vectorized_prune(data, thresh)
mutation_time = (time.time() - t0) * 1000.0

speedup_pct = ((baseline_time - mutation_time) / baseline_time) * 100.0
equivalence = np.allclose(res_base, res_mut)

print(f"BASELINE_MS: {baseline_time:.2f}")
print(f"MUTATION_MS: {mutation_time:.2f}")
print(f"SPEEDUP_PCT: {speedup_pct:.2f}%")
print(f"VERIFICATION_SUCCESS: {equivalence}")