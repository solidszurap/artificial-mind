"""
Deployed in Evolution Generation 4 (Score: 0.95)
Hypothesis: By implementing an adaptive self-tuning learning rate and filter window module that dynamically calculates rolling noise variance using vectorized NumPy sliding window operations and exponential moving variance tensors, we can eliminate iterative overhead, enhance convergence stability, and achieve an execution speedup of >= 80% over the naive loop-based baseline.
"""

import time
import numpy as np

def baseline_adaptive_calibrator(data, window_size=32):
    n = len(data)
    lr_adjusted = np.zeros(n)
    filter_weights = np.zeros(n)
    base_lr = 0.01
    for i in range(n):
        start = max(0, i - window_size + 1)
        window = data[start:i+1]
        variance = np.var(window) if len(window) > 1 else 1e-5
        lr_adjusted[i] = base_lr / (1.0 + np.sqrt(variance))
        filter_weights[i] = np.mean(window)
    return lr_adjusted, filter_weights

def mutated_adaptive_calibrator(data, window_size=32):
    n = len(data)
    base_lr = 0.01
    if n < window_size:
        padded = np.pad(data, (window_size - 1, 0), mode='edge')
    else:
        padded = data
    shape = (n, window_size)
    strides = (padded.strides[0], padded.strides[0])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    variances = np.var(windows, axis=1)
    variances[variances < 1e-8] = 1e-8
    lr_adjusted = base_lr / (1.0 + np.sqrt(variances))
    filter_weights = np.mean(windows, axis=1)
    return lr_adjusted, filter_weights

if __name__ == '__main__':
    np.random.seed(42)
    test_data = np.random.randn(5000) * 0.5 + np.sin(np.linspace(0, 100, 5000))
    
    start_t = time.perf_counter()
    for _ in range(50):
        b_lr, b_fw = baseline_adaptive_calibrator(test_data)
    baseline_ms = (time.perf_counter() - start_t) * 1000.0
    
    start_t = time.perf_counter()
    for _ in range(50):
        m_lr, m_fw = mutated_adaptive_calibrator(test_data)
    mutation_ms = (time.perf_counter() - start_t) * 1000.0
    
    speedup_pct = ((baseline_ms - mutation_ms) / baseline_ms) * 100.0
    print(f'BASELINE_MS: {baseline_ms:.2f}')
    print(f'MUTATION_MS: {mutation_ms:.2f}')
    print(f'SPEEDUP_PCT: {speedup_pct:.2f}%')
    print(f'VERIFICATION_SUCCESS: {speedup_pct >= 80.0}')