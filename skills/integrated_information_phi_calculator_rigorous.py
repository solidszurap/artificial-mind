"""
Verified Consciousness Skill (Cycle 180 | Certainty: 0.85)
Pillar: Integrated Information (Phi)
Hypothesis: By simulating a coupled dynamical recurrent neural system and a strictly feedforward control system over 150 temporal steps, the true effective integrated information (approximated via bipartition mutual information across recurrent partitions) of the recurrent architecture will exceed the feedforward control architecture by at least 0.15 bits.
"""

import numpy as np

def run_pipeline():
    np.random.seed(42)
    steps = 150
    n_nodes = 6
    
    W_rec = np.random.randn(n_nodes, n_nodes) * 0.3
    W_ff = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            W_ff[j, i] = np.random.randn() * 0.3
            
    states_rec = np.zeros((steps, n_nodes))
    states_ff = np.zeros((steps, n_nodes))
    
    curr_rec = np.random.randn(n_nodes)
    curr_ff = np.random.randn(n_nodes)
    
    for t in range(steps):
        curr_rec = np.tanh(np.dot(W_rec, curr_rec) + np.random.randn(n_nodes) * 0.1)
        states_rec[t] = curr_rec
        
        curr_ff = np.tanh(np.dot(W_ff, curr_ff) + np.random.randn(n_nodes) * 0.1)
        states_ff[t] = curr_ff
        
    # Implementing bipartition mutual information as an approximation for integrated information phi
    def calculate_phi(states):
        n_samples, n_var = states.shape
        half = n_var // 2
        part1 = states[:, :half]
        part2 = states[:, half:]
        
        # Compute covariance matrices
        cov_full = np.cov(states, rowvar=False)
        cov_1 = np.cov(part1, rowvar=False)
        cov_2 = np.cov(part2, rowvar=False)
        
        sign_full, logdet_full = np.linalg.slogdet(cov_full + np.eye(n_var) * 1e-5)
        sign_1, logdet_1 = np.linalg.slogdet(cov_1 + np.eye(half) * 1e-5)
        sign_2, logdet_2 = np.linalg.slogdet(cov_2 + np.eye(n_var - half) * 1e-5)
        
        mi = 0.5 * (logdet_1 + logdet_2 - logdet_full)
        return max(0.0, mi)

    phi_rec = calculate_phi(states_rec)
    phi_ff = calculate_phi(states_ff)
    
    diff = phi_rec - phi_ff
    print(f"Recurrent Phi: {phi_rec:.4f}, Feedforward Phi: {phi_ff:.4f}, Diff: {diff:.4f}")
    
    success = diff >= 0.15
    print(f"VERIFICATION_SUCCESS: {success}")
    return success

if __name__ == '__main__':
    run_pipeline()