"""
Deployed in Evolution Generation 7 (Score: 0.95)
Hypothesis: By implementing a vectorized Run-Length Encoding (RLE) compressor using NumPy contiguous array slicing and shift-based boundary detection (np.diff), we can compress repetitive state logs into compact run-value pairs in a single pass without iterative Python loops, reducing telemetry disk I/O and memory footprint by >= 50% compared to a naive Python loop implementation.
"""

import time
import numpy as np

def baseline_rle(data):
    if len(data) == 0:
        return [], []
    encoded_values = []
    encoded_counts = []
    current_val = data[0]
    current_count = 1
    for val in data[1:]:
        if val == current_val:
            current_count += 1
        else:
            encoded_values.append(current_val)
            encoded_counts.append(current_count)
            current_val = val
            current_count = 1
    encoded_values.append(current_val)
    encoded_counts.append(current_count)
    return np.array(encoded_values), np.array(encoded_counts)

def vectorized_rle(data):
    if data.size == 0:
        return np.array([], dtype=data.dtype), np.array([], dtype=int)
    
    # Find indices where values change
    change_indices = np.concatenate(([0], np.where(data[:-1] != data[1:])[0] + 1, [len(data)]))
    encoded_values = data[change_indices[:-1]]
    encoded_counts = np.diff(change_indices)
    return encoded_values, encoded_counts

if __name__ == '__main__':
    # Simulate telemetry state logs with repeating patterns
    np.random.seed(42)
    base_pattern = np.random.randint(0, 5, size=200)
    data = np.repeat(base_pattern, np.random.randint(1, 10, size=200))
    
    # Benchmark Baseline
    start = time.perf_counter()
    for _ in range(1000):
        b_vals, b_counts = baseline_rle(data)
    baseline_ms = (time.perf_counter() - start) * 1000.0
    
    # Benchmark Mutation
    start = time.perf_counter()
    for _ in range(1000):
        m_vals, m_counts = vectorized_rle(data)
    mutation_ms = (time.perf_counter() - start) * 1000.0
    
    speedup_pct = ((baseline_ms - mutation_ms) / baseline_ms) * 100.0
    
    # Verification
    valid = np.array_equal(b_vals, m_vals) and np.array_equal(b_counts, m_counts)
    
    print(f'BASELINE_MS: {baseline_ms:.2f}')
    print(f'MUTATION_MS: {mutation_ms:.2f}')
    print(f'SPEEDUP_PCT: {speedup_pct:.2f}%')
    print(f'VERIFICATION_SUCCESS: {valid}')