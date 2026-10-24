"""
Deployed in Evolution Generation 9 (Score: 0.95)
Hypothesis: By replacing Python-loop-based conditional entropy calculations with a fully vectorized NumPy contingency matrix formulation using np.bincount with multidimensional encoding and broadcasting, we can eliminate nested python loops over states, reducing calculation time by >= 40% while ensuring exact numerical equivalence.
"""

import time
import numpy as np

def baseline_markov_entropy(states, actions, next_states):
    n = len(states)
    joint_counts = {}
    state_counts = {}
    for s, a, ns in zip(states, actions, next_states):
        key = (s, a, ns)
        joint_counts[key] = joint_counts.get(key, 0) + 1
        state_counts[s] = state_counts.get(s, 0) + 1
    
    entropy = 0.0
    total = float(n)
    for (s, a, ns), count in joint_counts.items():
        p_joint = count / total
        p_s = state_counts[s] / total
        p_conditional = count / state_counts[s]
        entropy -= p_joint * np.log2(p_conditional + 1e-12)
    return entropy

def mutated_markov_entropy(states, actions, next_states):
    states = np.asarray(states, dtype=np.int64)
    actions = np.asarray(actions, dtype=np.int64)
    next_states = np.asarray(next_states, dtype=np.int64)
    
    max_s = max(np.max(states), 1) + 1
    max_a = max(np.max(actions), 1) + 1
    max_ns = max(np.max(next_states), 1) + 1
    
    encoded_joint = states * (max_a * max_ns) + actions * max_ns + next_states
    encoded_state = states
    
    joint_counts = np.bincount(encoded_joint)
    state_counts = np.bincount(encoded_state)
    
    total = float(len(states))
    
    valid_joints = np.nonzero(joint_counts)[0]
    if len(valid_joints) == 0:
        return 0.0
        
    s_vals = valid_joints // (max_a * max_ns)
    remainder = valid_joints % (max_a * max_ns)
    a_vals = remainder // max_ns
    ns_vals = remainder % max_ns
    
    c_joint = joint_counts[valid_joints].astype(np.float64)
    c_state = state_counts[s_vals].astype(np.float64)
    
    p_joint = c_joint / total
    p_conditional = c_joint / (c_state + 1e-12)
    
    entropy = -np.sum(p_joint * np.log2(p_conditional + 1e-12))
    return float(entropy)

np.random.seed(42)
n_samples = 5000
states = np.random.randint(0, 10, size=n_samples)
actions = np.random.randint(0, 4, size=n_samples)
next_states = np.random.randint(0, 10, size=n_samples)

t0 = time.time()
base_res = baseline_markov_entropy(states, actions, next_states)
t1 = time.time()
baseline_ms = (t1 - t0) * 1000.0

t0 = time.time()
mut_res = mutated_markov_entropy(states, actions, next_states)
t1 = time.time()
mutation_ms = (t1 - t0) * 1000.0

speedup_pct = ((baseline_ms - mutation_ms) / baseline_ms) * 100.0
verification_success = bool(np.isclose(base_res, mut_res, atol=1e-5))

print(f"BASELINE_MS: {baseline_ms:.2f}")
print(f"MUTATION_MS: {mutation_ms:.2f}")
print(f"SPEEDUP_PCT: {speedup_pct:.2f}%")
print(f"VERIFICATION_SUCCESS: {verification_success}")