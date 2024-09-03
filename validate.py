import numpy as np


def never_enter_bad_prob(transition_matrix, initial_state, good_states, k):
    # Make bad states absorbing
    for i in range(len(transition_matrix)):
        if i not in good_states:
            transition_matrix[i, :] = 0
            transition_matrix[i, i] = 1

    # Calculate the k-th power of the transition matrix
    k_step_matrix = np.linalg.matrix_power(transition_matrix, k)
    print(k_step_matrix)
    # Extract the probability of remaining in good states from the initial state
    probabilities_after_k = k_step_matrix[initial_state, :]

    probability_never_bad = sum(probabilities_after_k[state] for state in good_states)

    return probability_never_bad


def monte_carlo_simulation(transition_matrix, initial_state, good_states, k, num_simulations=10000):
    count_stayed_good = 0  # Count of paths that stayed in good states

    for _ in range(num_simulations):
        current_state = initial_state
        stayed_good = True

        for step in range(k):
            next_state = np.random.choice(range(len(transition_matrix)), p=transition_matrix[current_state])
            if next_state not in good_states:
                stayed_good = False
                break  # Exit the loop if a bad state is entered
            current_state = next_state

        if stayed_good:
            count_stayed_good += 1  # Increment count if the path stayed in good states

    return count_stayed_good / num_simulations  # Estimated probability


def steady_state_probabilities(P):
    # Get the number of states
    num_states = P.shape[0]

    # Create a matrix A and vector b for the system of linear equations
    A = np.vstack((P.T - np.eye(num_states), np.ones(num_states)))
    b = np.vstack((np.zeros((num_states, 1)), np.ones((1, 1))))

    # Solve the system of linear equations
    pi = np.linalg.lstsq(A, b, rcond=None)[0]

    # Normalize the solution to ensure the probabilities sum to 1
    pi = pi / np.sum(pi)

    return pi.flatten()


# Define your transition matrix and the good states
transition_matrix = np.array([
    [0.4, 0.3, 0.2, 0.1],
    [0.2,   0.5,  0.2,   0.1],
    [0.1,  0.3,   0.4,  0.2],
    [0.1, 0.2, 0.3,  0.4]
])
# transition_matrix = np.array([[0.7, 0.3],
#                               [0.29, 0.71]])
good_states = [0, 2]  # Assuming state 2 is bad
initial_state = 0  # Starting from state 0


# for k in range(1, 20):
# Calculate the probability after k steps
k = 1
probability = never_enter_bad_prob(transition_matrix, initial_state, good_states, k)
print(f"Probability of not entering a bad state after {k} steps: {probability}")

print(steady_state_probabilities(transition_matrix))

# # Monte Carlo simulation
num_simulations = 100000  # Number of paths to simulate
estimated_probability = monte_carlo_simulation(transition_matrix, initial_state, good_states, k, num_simulations)
print(f"Monte Carlo estimated probability: {estimated_probability:.8f}")
