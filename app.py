import numpy as np
from typedefs import *
import random


class App:
    def __init__(self, trial, id, links, T, max_n_states, max_n_flows, max_n_flow_hop):
        random.seed(123+trial+id)
        np.random.seed(123+trial+id)
        self.id = id
        self.links = links
        self.T = T
        self.max_n_states = max_n_states
        self.max_n_flows = max_n_flows
        self.max_n_flow_hop = max_n_flow_hop
        self.n_states = random.randint(2, self.max_n_states)
        self.states = self.generate_states()
        self.transitions = self.generate_transitions()
        self.k_max = self.determine_k_max()

    def generate_states(self):
        states = []
        for s in range(0, self.n_states):
            state = State(s)

            for f in range(random.randint(1, self.max_n_flows)):
                flow = Flow(f, random.sample(self.links, random.randint(1,
                            self.max_n_flow_hop)), int(random.choice([p for p in range(self.max_n_flow_hop*2, self.T + 1) if self.T % p == 0])))
                state.flows.append(flow)
            states += [state]
        return states

    def generate_transitions(self):
        n_states = len(self.states)
        transitions = np.zeros((n_states, n_states))
        for i in range(n_states):
            # Generate n-1 random numbers and sort them
            cut_points = np.sort(np.random.rand(n_states - 1))

            # Calculate differences between successive numbers to simulate the random probabilities
            # Append 1 at the end to ensure the last segment is included
            probs = np.diff(np.hstack(([0], cut_points, [1])))

            # Ensure the sum is exactly 1 by adjusting the last element
            probs[-1] = 1 - probs[:-1].sum()

            # Store the probabilities
            transitions[i] = probs

        for row in transitions:
            assert sum(row) == 1
            for i in row:
                assert i > 0
        return transitions

    def determine_k_max(self):
        self.steady = self.steady_state_probabilities()
        self.M_k = [self.transitions[0]]
        for k in range(1, 100):
            p = np.linalg.matrix_power(self.transitions, k)[0]
            self.M_k += [p]
            if np.abs(p[0] - self.steady[0]) <= 1e-4:
                return k

    def steady_state_probabilities(self):
        # Create a matrix A and vector b for the system of linear equations
        A = np.vstack((self.transitions.T - np.eye(self.n_states), np.ones(self.n_states)))
        b = np.vstack((np.zeros((self.n_states, 1)), np.ones((1, 1))))

        # Solve the system of linear equations
        pi = np.linalg.lstsq(A, b, rcond=None)[0]

        # Normalize the solution to ensure the probabilities sum to 1
        pi = pi / np.sum(pi)

        return pi.flatten()
