"""
Deployed in Evolution Generation 23 (Score: 0.95)
Hypothesis: By optimizing the existing vectorized cosine similarity matrix module using in-place L2 norm normalization combined with optimized matrix multiplication (np.dot or @ operator) and avoiding redundant intermediate allocations, we can achieve even higher computation speedups and lower memory consumption for high-dimensional state embedding comparisons.
"""

import time
import numpy as np

def baseline_cosine_similarity(A, B):
    norms_a = np.linalg.norm(A, axis=1, keepdims=True)
    norms_b = np.linalg.norm(B, axis=1, keepdims=True)
    dot_prod = np.dot(A, B.T)
    eps = 1e-8
    return dot_prod / (np.dot(norms_a, norms_b.T) + eps)

def mutated_cosine_similarity(A, B):
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return np.dot(A_norm, B_norm.T)

if __name__ == '__main__':
    np.random.seed(42)
    A = np.random.randn(2000, 128)
    B = np.random.randn(2000, 128)
    
    start = time.perf_counter()
    for _ in range(10):
        res_base = baseline_cosine_similarity(A, B)
    baseline_ms = (time.perf_counter() - start) * 1000.0
    
    start = time.perf_counter()
    for _ in range(10):
        res_mut = mutated_cosine_similarity(A, B)
    mutation_ms = (time.perf_counter() - start) * 1000.0
    
    speedup_pct = ((baseline_ms - mutation_ms) / baseline_ms) * 100.0
    verification = np.allclose(res_base, res_mut, atol=1e-5)
    
    print(f'BASELINE_MS: {baseline_ms:.2f}')
    print(f'MUTATION_MS: {mutation_ms:.2f}')
    print(f'SPEEDUP_PCT: {speedup_pct:.2f}%')
    print(f'VERIFICATION_SUCCESS: {verification}')