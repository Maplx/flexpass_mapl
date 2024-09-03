import numpy as np

transition_matrix = np.array([
    [0.4, 0.3, 0.2, 0.1],
    [0.2,   0.5,  0.2,   0.1],
    [0.1,  0.3,   0.4,  0.2],
    [0.1, 0.2, 0.3,  0.4]
])


def steady_state_probabilities(M):
    # Get the number of states
    num_states = M.shape[0]

    # Create a matrix A and vector b for the system of linear equations
    A = np.vstack((M.T - np.eye(num_states), np.ones(num_states)))
    b = np.vstack((np.zeros((num_states, 1)), np.ones((1, 1))))

    # Solve the system of linear equations
    pi = np.linalg.lstsq(A, b, rcond=None)[0]

    # Normalize the solution to ensure the probabilities sum to 1
    pi = pi / np.sum(pi)

    return pi.flatten()


def k_th_power(M, k):
    return np.linalg.matrix_power(M, k)


s = steady_state_probabilities(transition_matrix)
print(s)
for k in range(1, 100):
    p = k_th_power(transition_matrix, k)[0]
    print(k, p)
    if np.abs(p[0] - s[0]) <= 1e-6:
        break
