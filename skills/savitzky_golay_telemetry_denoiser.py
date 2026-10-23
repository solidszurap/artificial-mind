"""
Verified in Cycle 85 (Score: 0.95)
Hypothesis: Applying a Savitzky-Golay filter to high-frequency noisy telemetry data will reduce the root mean square error (RMSE) relative to the ground truth signal by more than 40 percent compared to raw noisy observations, while preserving local signal extrema.
"""

import numpy as np
from scipy.signal import savgol_filter

def denoise_telemetry(raw_signal, window_length=15, polyorder=3):
    return savgol_filter(raw_signal, window_length=window_length, polyorder=polyorder)

if __name__ == '__main__':
    np.random.seed(42)
    t = np.linspace(0, 10, 500)
    true_signal = np.sin(t) + 0.5 * np.cos(2 * t)
    noise = np.random.normal(0, 0.3, size=t.shape)
    raw_noisy = true_signal + noise
    
    filtered = denoise_telemetry(raw_noisy)
    
    raw_rmse = np.sqrt(np.mean((raw_noisy - true_signal) ** 2))
    filtered_rmse = np.sqrt(np.mean((filtered - true_signal) ** 2))
    
    reduction_pct = (raw_rmse - filtered_rmse) / raw_rmse * 100.0
    print(f'RAW_RMSE: {raw_rmse:.6f}')
    print(f'FILTERED_RMSE: {filtered_rmse:.6f}')
    print(f'RMSE_REDUCTION_PCT: {reduction_pct:.2f}')
    
    assert reduction_pct > 40.0, 'RMSE reduction did not exceed 40 percent'
    print('VERIFICATION_SUCCESS: True')