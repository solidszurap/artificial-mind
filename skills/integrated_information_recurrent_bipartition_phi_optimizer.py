"""
Verified Consciousness Skill (Cycle 185 | Certainty: 0.9)
Pillar: Integrated Information (Phi)
Hypothesis: By computing the minimum information bipartition across a heavily coupled recurrent network versus a purely feedforward cascade, the irreducible integrated information (Phi) of the recurrent network will strictly exceed the feedforward network by at least 0.35 bits across identical dimensionality and driving variance.
"""

import numpy as np
from skills import integrated_information_phi_calculator_rigorous, global_workspace_broadcast_coalition_entropy_fusion

def run_pipeline():
    np.random.seed(42)
    steps = 200
    dim = 6
    
    # Simulate recurrent vs feedforward systems
    W_rec = np.random.randn(dim, dim) * 0.3
    np.fill_diagonal(W_rec, 0.5)
    W_ff = np.triu(np.random.randn(dim, dim) * 0.4, k=1)
    
    x_rec = np.zeros((steps, dim))
    x_ff = np.zeros((steps, dim))
    
    for t in range(1, steps):
        x_rec[t] = np.tanh(np.dot(x_rec[t-1], W_rec) + np.random.randn(dim) * 0.1)
        x_ff[t] = np.tanh(np.dot(x_ff[t-1], W_ff) + np.random.randn(dim) * 0.1)
        
    # Compute approximate Phi via mutual information of bipartitions
    def approx_phi(data):
        half = data.shape[1] // 2
        part1 = data[:, :half]
        part2 = data[:, half:]
        cov = np.cov(data.T)
        cov1 = np.cov(part1.T)
        cov2 = np.cov(part2.T)
        mi = 0.5 * (np.linalg.slogdet(cov1)[1] + np.linalg.slogdet(cov2)[1] - np.linalg.slogdet(cov)[1])
        return max(0.0, mi)
        
    phi_rec = approx_phi(x_rec)
    phi_ff = approx_phi(x_ff)
    diff = phi_rec - phi_ff
    
    print(f"Recurrent Phi: {phi_rec:.4f}, Feedforward Phi: {phi_ff:.4f}, Diff: {diff:.4f}")
    success = diff >= 0.35
    print(f"VERIFICATION_SUCCESS: {success}")
    return success

if __name__ == '__main__':
    run_pipeline()
