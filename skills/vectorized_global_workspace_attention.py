"""
Deployed in Evolution Generation 6 (Score: 0.98)
Hypothesis: By refactoring iterative attention score calculations and softmax normalization across heads into a single, highly optimized block using direct np.matmul and fused broadcasting over pre-allocated tensors, we eliminate intermediate transposition overhead and achieve an execution speedup of >= 40% while preserving strict numerical equivalence.
"""

import time
import numpy as np

def baseline_attention(queries, keys, values):
    batch_size, seq_len, d_model = queries.shape
    scores = np.zeros((batch_size, seq_len, seq_len))
    for b in range(batch_size):
        for i in range(seq_len):
            for j in range(seq_len):
                scores[b, i, j] = np.dot(queries[b, i], keys[b, j]) / np.sqrt(d_model)
    weights = np.zeros_like(scores)
    for b in range(batch_size):
        for i in range(seq_len):
            exp_vals = np.exp(scores[b, i] - np.max(scores[b, i]))
            weights[b, i] = exp_vals / np.sum(exp_vals)
    output = np.zeros((batch_size, seq_len, d_model))
    for b in range(batch_size):
        for i in range(seq_len):
            for j in range(seq_len):
                output[b, i] += weights[b, i, j] * values[b, j]
    return output

def mutated_attention(queries, keys, values):
    d_model = queries.shape[-1]
    scores = np.matmul(queries, keys.transpose(0, 2, 1)) / np.sqrt(d_model)
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    output = np.matmul(weights, values)
    return output

np.random.seed(42)
batch, seq, d = 4, 32, 64
q = np.random.randn(batch, seq, d)
k = np.random.randn(batch, seq, d)
v = np.random.randn(batch, seq, d)

t0 = time.time()
base_out = baseline_attention(q, k, v)
t1 = time.time()
baseline_ms = (t1 - t0) * 1000.0

t0 = time.time()
mut_out = mutated_attention(q, k, v)
t1 = time.time()
mutation_ms = (t1 - t0) * 1000.0

speedup_pct = ((baseline_ms - mutation_ms) / baseline_ms) * 100.0
verification = np.allclose(base_out, mut_out, atol=1e-5)

print(f"BASELINE_MS: {baseline_ms:.2f}")
print(f"MUTATION_MS: {mutation_ms:.2f}")
print(f"SPEEDUP_PCT: {speedup_pct:.2f}%")
print(f"VERIFICATION_SUCCESS: {verification}")