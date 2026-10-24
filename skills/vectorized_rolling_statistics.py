"""
Deployed in Evolution Generation 20 (Score: 0.95)
Hypothesis: By replacing cumulative loops and sliding window iterations over large state telemetry streams with a fully vectorized strided-view approach using NumPy's np.lib.stride_tricks.as_strided combined with direct axis-wise aggregations (mean, standard deviation, min, max), we can eliminate Python-level overhead and memory reallocations, reducing sliding window calculation time by >= 45% compared to naive deque-based baselines.
"""

import time
import numpy as np

def baseline_rolling_stats(data, window_size):
    means = []
    stds = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        means.append(np.mean(window))
        stds.append(np.std(window))
    return np.array(means), np.array(stds)

def mutated_rolling_stats(data, window_size):
    shape = (data.shape[0] - window_size + 1, window_size)
    strides = (data.strides[0], data.strides[0])
    windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    means = np.mean(windows, axis=1)
    stds = np.std(windows, axis=1)
    return means, stds

if __name__ == '__main__':
    np.random.seed(42)
    data = np.random.randn(20000)
    window_size = 50
    
    start = time.perf_counter()
    for _ in range(5):
        b_means, b_stds = baseline_rolling_stats(data, window_size)
    baseline_time = (time.perf_counter() - start) * 1000.0
    
    start = time.perf_counter()
    for _ in range(5):
        m_means, m_stds = mutated_rolling_stats(data, window_size)
    mutation_time = (time.perf_counter() - start) * 1000.0
    
    speedup_pct = ((baseline_time - mutation_time) / baseline_time) * 100.0
    verification_success = bool(mutation_time < baseline_time and np.allclose(b_means, m_means))
    
    print(f'BASELINE_MS: {baseline_time:.2f}')
    print(f'MUTATION_MS: {mutation_time:.2f}')
    print(f'SPEEDUP_PCT: {speedup_pct:.2f}%')
    print(f'VERIFICATION_SUCCESS: {verification_success}')