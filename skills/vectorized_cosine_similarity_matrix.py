"""
Deployed in Evolution Generation 14 (Score: 0.98)
Hypothesis: By replacing nested Python loops computing pairwise cosine similarity across high-dimensional embeddings with a fully vectorized matrix multiplication using dot products and L2 norm broadcasting in NumPy, we can eliminate all Python-level overhead and leverage optimized BLAS kernels, reducing execution time by >= 60% while maintaining exact numerical accuracy.
"""

import time
import numpy as np

def baseline_cosine_similarity(matrix_a, matrix_b):
    n, d = matrix_a.shape
    m, _ = matrix_b.shape
    sim = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dot = np.dot(matrix_a[i], matrix_b[j])
            norm_a = np.linalg.norm(matrix_a[i])
            norm_b = np.linalg.norm(matrix_b[j])
            if norm_a == 0 or norm_b == 0:
                sim[i, j] = 0.0
            else:
                sim[i, j] = dot / (norm_a * norm_b)
    return sim

def vectorized_cosine_similarity(matrix_a, matrix_b):
    norm_a = np.linalg.norm(matrix_a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(matrix_b, axis=1, keepdims=True)
    norm_a[norm_a == 0] = 1.0
    norm_b[norm_b == 0] = 1.0
    dot_product = np.dot(matrix_a, matrix_b.T)
    sim = dot_product / (norm_a * norm_b.T)
    return sim

if __name__ == '__main__':
    np.random.seed(42)
    mat_a = np.random.randn(200, 64)
    mat_b = np.random.randn(200, 64)

    start = time.time()
    for _ in range(3):
        res_base = baseline_cosine_similarity(mat_a, mat_b)
    baseline_ms = (time.time() - start) * 1000.0 / 3.0

    start = time.time()
    for _ in range(3):
        res_mut = vectorized_cosine_similarity(mat_a, mat_b)
    mutation_ms = (time.time() - start) * 1000.0 / 3.0

    speedup_pct = ((baseline_ms - mutation_ms) / baseline_ms) * 100.0
    verified = np.allclose(res_base, res_mut, atol=1e-5) and (mutation_ms < baseline_ms)

    print(f'BASELINE_MS: {baseline_ms:.2f}')
    print(f'MUTATION_MS: {mutation_ms:.2f}')
    print(f'SPEEDUP_PCT: {speedup_pct:.2f}%')
    print(f'VERIFICATION_SUCCESS: {verified}')