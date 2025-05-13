import importlib
import always_heu
import adjustment1trial
import always_heu_3
import setcove_bipartite
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar

# Reload modules
importlib.reload(always_heu)
importlib.reload(adjustment1trial)
importlib.reload(always_heu_3)
importlib.reload(setcove_bipartite)


from always_heu import Adjustment as Adjustment1
from always_heu_3 import Adjustment as Adjustment2
from adjustment1trial import Adjustment as Adjustment3
from setcove_bipartite import Adjustment as Adjustment4
import numpy as np

# Simulation parameters
n_apps_values = [4,6,8,10,12] # Varying number of applications
T = 50
links = range(30)
max_n_states = 10
max_n_flows = 8
max_n_flow_hop = 5

def save_shared_transitions(apps, steps=2000):
    transitions = []
    for app in apps:
        current = 0
        seq = []
        for _ in range(steps):
            next_state = np.random.choice(range(app.n_states), p=app.transitions[current])
            seq.append(next_state)
            current = next_state
        transitions.append(seq)
    np.save("shared_transitions.npy", transitions)


# Function to run simulation for a given Adjustment class
def run_simulation(AdjustmentClass, n_apps):
    adj = AdjustmentClass(trial=21, n_apps=n_apps, T=T, links=links,
                          max_n_states=max_n_states, max_n_flows=max_n_flows,
                          max_n_flow_hop=max_n_flow_hop, verbose=False)
    save_shared_transitions(adj.apps)
    
    results = adj.run()

    # Collect metrics
    flex_values = [res["flex"] for res in results]
    time_values = [res["time"] for res in results]
    reconfig_values = [res["reconfig_count"] for res in results]
    n_apps = [res["n_affected_apps"] for res in results]

    avg_flex = sum(flex_values) / len(flex_values) if flex_values else 0
    avg_time = sum(time_values) / len(time_values) if time_values else 0
    reconfig_count_last = reconfig_values[-1] if reconfig_values else 0
    avg_apps = sum(n_apps) / len(n_apps) if flex_values else 0

    return {
        "avg_flex": avg_flex,
        "avg_time": avg_time,
        "avg_apps": avg_apps,
        "reconfig_count_last": reconfig_count_last

    }

# Data storage for each method
results_heu = []
results_heu3 = []
results_adjust1 = []
results_setbip = []
# Run simulations across different n_apps values with a progress bar
for n_apps in tqdm(n_apps_values, desc="Running simulations for different n_apps"):
    adj = Adjustment1(trial=20, n_apps=n_apps, T=50, links=range(30),
                         max_n_states=10, max_n_flows=8, max_n_flow_hop=5,
                         verbose=False)
    save_shared_transitions(adj.apps)

    results_heu.append(run_simulation(Adjustment1, n_apps))
    results_heu3.append(run_simulation(Adjustment2, n_apps))

    results_adjust1.append(run_simulation(Adjustment3, n_apps))

    results_setbip.append(run_simulation(Adjustment4, n_apps))


# Plotting the results with markers and custom x-axis ticks
def plot_metric(metric, label):
    plt.figure()
    plt.plot(n_apps_values, [res[metric] for res in results_heu], label="SPaL", marker='o')
    plt.plot(n_apps_values, [res[metric] for res in results_heu3], label="Current_State", marker='o')
    plt.plot(n_apps_values, [res[metric] for res in results_adjust1], label="SP", marker='o')
    plt.plot(n_apps_values, [res[metric] for res in results_setbip], label="SP+CP", marker='o')
    plt.xlabel("Number of Applications (n_apps)")
    plt.ylabel(label)
    plt.xticks(n_apps_values)  # Set x-axis ticks to be exactly the n_apps values
    plt.title(f"{label} vs. Number of Applications")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)  # Optional: add grid lines for readability
    plt.show()

# Generate the three required plots

plot_metric("avg_flex", "Average Flexibility")
plot_metric("avg_apps", "Average Adjusted Partition Number")
plot_metric("reconfig_count_last", " Reconfiguration Count")