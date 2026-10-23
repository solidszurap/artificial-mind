"""
Verified in Cycle 88 (Score: 0.95)
Hypothesis: Applying a rolling window median absolute deviation (MAD) robust outlier removal algorithm to a noisy time-series containing injected sparse spikes will reduce the mean absolute error (MAE) relative to the ground truth by at least 30 percent compared to the raw noisy signal.
"""

import numpy as np

def clean_rolling_mad_outliers(series, window_size=50, threshold=3.5):
    n = len(series)
    cleaned = np.copy(series)
    half = window_size // 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window = series[start:end]
        med = np.median(window)
        mad = np.median(np.abs(window - med))
        if mad == 0:
            continue
        modified_z = 0.6745 * (series[i] - med) / mad
        if np.abs(modified_z) > threshold:
            cleaned[i] = med
    return cleaned

def run_benchmark():
    np.random.seed(42)
    t = np.linspace(0, 50, 1000)
    true_signal = np.sin(t)
    noise = np.random.normal(0, 0.1, len(t))
    raw_signal = true_signal + noise
    spike_indices = np.random.choice(len(t), size=20, replace=False)
    raw_signal[spike_indices] += np.random.choice([-1, 1], size=20) * 5.0
    
    cleaned_signal = clean_rolling_mad_outliers(raw_signal, window_size=31, threshold=3.0)
    
    raw_mae = np.mean(np.abs(raw_signal - true_signal))
    cleaned_mae = np.mean(np.abs(cleaned_signal - true_signal))
    mae_reduction_pct = (raw_mae - cleaned_mae) / raw_mae * 100.0
    
    print(f'RAW_MAE: {raw_mae:.6f}')
    print(f'CLEANED_MAE: {cleaned_mae:.6f}')
    print(f'MAE_REDUCTION_PCT: {mae_reduction_pct:.2f}')
    
    success = cleaned_mae < raw_mae * 0.70
    print(f'VERIFICATION_SUCCESS: {success}')
    assert success, 'MAE reduction did not meet the 30 percent threshold'

if __name__ == '__main__':
    run_benchmark()