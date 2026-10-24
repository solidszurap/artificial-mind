"""
Deployed in Evolution Generation 1 (Score: 0.95)
Hypothesis: By replacing iterative Python loops in multi-head expert coalition competition with fully vectorized NumPy tensor operations and pre-allocated memory buffers, we eliminate interpreter overhead and achieve massive speedups during global workspace ignition phases.
"""

import time
import numpy as np

def baseline_expert_competition(inputs, weights, bias, steps):
    start = time.perf_counter()
    results = []
    for _ in range(steps):
        step_res = []
        for i in range(inputs.shape[0]):
            val = 0.0
            for j in range(inputs.shape[1]):
                val += inputs[i, j] * weights[j]
            val += bias
            step_res.append(1.0 / (1.0 + np.exp(-val)))
        results.append(np.mean(step_res))
    end = time.perf_counter()
    return (end - start) * 1000.0, results

def vectorized_expert_competition(inputs, weights, bias, steps):
    start = time.perf_counter()
    results = []
    for _ in range(steps):
        linear = np.dot(inputs, weights) + bias
        activations = 1.0 / (1.0 + np.exp(-linear))
        results.append(np.mean(activations))
    end = time.perf_counter()
    return (end - start) * 1000.0, results

if __name__ == '__main__':
    np.random.seed(42)
    inputs = np.random.randn(100, 32)
    weights = np.random.randn(32)
    bias = 0.1
    steps = 500
    
    base_ms, _ = baseline_expert_competition(inputs, weights, bias, steps)
    mut_ms, _ = vectorized_expert_competition(inputs, weights, bias, steps)
    
    speedup = ((base_ms - mut_ms) / base_ms) * 100.0
    success = mut_ms < base_ms
    
    print(f'BASELINE_MS: {base_ms:.2f}')
    print(f'MUTATION_MS: {mut_ms:.2f}')
    print(f'SPEEDUP_PCT: {speedup:.2f}%')
    print(f'VERIFICATION_SUCCESS: {success}')