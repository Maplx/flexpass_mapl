import numpy as np
import matplotlib.pyplot as plt

# Transition matrix
T = np.array([
    [0.2, 0.8, 0.0, 0.0],
    [0.0, 0.0, 0.4, 0.6],
    [0.0, 0.0, 0.1, 0.9],
    [0.2, 0.1, 0.7, 0.0]
])

# Adjusted code to include the initial state probabilities (π0)
k_max = 20

def k_step_success_prob_with_pi0(T, feasible_states, k_max, pi_0):
    n_states = T.shape[0]
    modified_T = T.copy()
    
    # Modify the transition matrix to make infeasible states absorbing states
    for i in range(n_states):
        if i not in feasible_states:
            modified_T[i, :] = 0
            modified_T[i, i] = 1  # Absorbing state
    
    # Compute k-step success probabilities starting from the initial state distribution
    success_probs = []
    current_pi = pi_0
    for k in range(1, k_max + 1):
        current_pi = np.dot(current_pi, modified_T)  # Update the probability distribution
        success_prob = np.sum([current_pi[i] for i in feasible_states])  # Sum probabilities of feasible states
        success_probs.append(success_prob)
    
    return success_probs

# Define the combinations of feasible states
state_combinations = [
    [0], [1], [2], [3],
    [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
    [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3],
    [0, 1, 2, 3]
]

# Initial state probabilities (starting with a custom distribution)
pi_0 = np.array([0.7,0.2,0,0.1])


# Calculate k-step success probabilities for each combination with the initial state distribution
results_with_pi0 = {tuple(states): k_step_success_prob_with_pi0(T, states, k_max, pi_0) for states in state_combinations}

# Plotting the results
plt.figure(figsize=(12, 8))
for states, probs in results_with_pi0.items():
    plt.plot(range(1, k_max + 1), probs, label=str(states), marker='o', markersize=6, markerfacecolor='none')

plt.title("k-Step Success Probabilities with Initial State Distribution")
plt.xlabel("k (steps)")
plt.ylabel("Success Probability")
plt.legend(title="Feasible States", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
