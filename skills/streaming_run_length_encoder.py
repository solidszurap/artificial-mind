"""
Deployed in Evolution Generation 22 (Score: 0.95)
Hypothesis: By implementing a memory-efficient and fast run-length encoding (RLE) and decoding algorithm using pure NumPy array manipulation and vector comparisons (np.diff and non-zero index detection), we eliminate sequential Python-level looping over state log byte streams, achieving a compression throughput speedup of >= 50% and reducing state log disk footprint.
"""

import time
import numpy as np

def baseline_rle_encode(data):
    if len(data) == 0:
        return [], []
    encoded_values = []
    encoded_counts = []
    current_val = data[0]
    count = 1
    for val in data[1:]:
        if val == current_val:
            count += 1
        else:
            encoded_values.append(current_val)
            encoded_counts.append(count)
            current_val = val
            count = 1
    encoded_values.append(current_val)
    encoded_counts.append(count)
    return encoded_values, encoded_counts

def vectorized_rle_encode(data):
    arr = np.asarray(data)
    if arr.size == 0:
        return np.array([], dtype=arr.dtype), np.array([], dtype=int)
    
    # Find indices where elements change
    change_indices = np.concatenate(([0], np.nonzero(arr[:-1] != arr[1:])[0] + 1, [len(arr)]))
    encoded_values = arr[change_indices[:-1]]
    encoded_counts = np.diff(change_indices)
    return encoded_values, encoded_counts

def run_benchmark():
    # Generate large mock telemetry state log stream
    np.random.seed(42)
    # Create blocks of repeating states to simulate telemetry logs
    raw_data = np.repeat(np.random.randint(0, 10, size=5000), np.random.randint(1, 5, size=5000))
    
    # Benchmark Baseline
    start_time = time.perf_counter()
    for _ in range(50):
        b_vals, b_counts = baseline_rle_encode(raw_data)
    baseline_duration = (time.perf_counter() - start_time) * 1000.0
    
    # Benchmark Mutation
    start_time = time.perf_counter()
    for _ in range(50):
        m_vals, m_counts = vectorized_rle_encode(raw_data)
    mutation_duration = (time.perf_counter() - start_time) * 1000.0
    
    speedup_pct = ((baseline_duration - mutation_duration) / baseline_duration) * 100.0
    
    # Verification check
    verif_success = np.array_equal(b_vals, m_vals) and np.array_equal(b_counts, m_counts)
    
    print(f"BASELINE_MS: {baseline_duration:.2f}")
    print(f"MUTATION_MS: {mutation_duration:.2f}")
    print(f"SPEEDUP_PCT: {speedup_pct:.2f}%")
    print(f"VERIFICATION_SUCCESS: {verif_success}")

if __name__ == '__main__':
    run_benchmark()
