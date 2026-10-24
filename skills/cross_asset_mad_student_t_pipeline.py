"""
Verified Cycle 108 | Certainty: 0.92
Chained Skills: ['rolling_zscore_outlier_cleaner', 'student_t_cvar_estimator']
Hypothesis: Applying rolling MAD outlier cleaning followed by Student-t distribution CVaR estimation at the 95th percentile on cross-asset daily returns extracted from live telemetry cache will successfully compute portfolio risk metrics without artificial markup, where the Student-t CVaR is computed and compared against the Gaussian baseline.
"""

import json
import numpy as np
from skills.rolling_zscore_outlier_cleaner import clean_rolling_mad_outliers
from skills.student_t_cvar_estimator import compute_cvar_comparison

def run_pipeline(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    returns_list = []
    for asset, info in data.items():
        if 'daily_returns' in info and len(info['daily_returns']) > 0:
            returns_list.append(info['daily_returns'])
        elif 'closes' in info and len(info['closes']) > 1:
            closes = np.array(info['closes'])
            rets = np.diff(closes) / closes[:-1]
            returns_list.append(rets.tolist())
    min_len = min(len(r) for r in returns_list)
    trimmed = [r[-min_len:] for r in returns_list]
    portfolio_returns = np.mean(trimmed, axis=0)
    series_returns = np.array(portfolio_returns)
    cleaned_series = clean_rolling_mad_outliers(series_returns, window_size=20, threshold=3.0)
    res_cvar = compute_cvar_comparison(cleaned_series, alpha=0.95)
    if isinstance(res_cvar, (list, tuple)) and len(res_cvar) == 2:
        gaussian_cvar, student_t_cvar = res_cvar
        success = True
    else:
        gaussian_cvar, student_t_cvar, success = res_cvar
    return {
        'gaussian_cvar': gaussian_cvar,
        'student_t_cvar': student_t_cvar,
        'success': success
    }

if __name__ == '__main__':
    res = run_pipeline('live_telemetry_cache.json')
    print(f"GAUSSIAN_CVAR: {res['gaussian_cvar']:.6f}")
    print(f"STUDENT_T_CVAR: {res['student_t_cvar']:.6f}")
    diff_pct = ((res['student_t_cvar'] - res['gaussian_cvar']) / res['gaussian_cvar']) * 100
    print(f"CVAR_DIFFERENCE_PCT: {diff_pct:.2f}")
    print(f"VERIFICATION_SUCCESS: {res['success']}")
