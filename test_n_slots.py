import importlib
import always_heu
import adjustment1trial
import bipartite_adjustment
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar

# Reload modules
importlib.reload(always_heu)
importlib.reload(adjustment1trial)
importlib.reload(bipartite_adjustment)

from always_heu import Adjustment as Adjustment1
from adjustment1trial import Adjustment as Adjustment2
from bipartite_adjustment import Adjustment as Adjustment3

# Simulation parameters
T_values = [30, 50, 70, 90, 110]  # Varying number of slots
n_apps = 5  # Fixed number of applications (can adjust as needed)
links = range(30)
max_n_states = 10
max_n_flows = 8
max_n_flow_hop = 5

# Function to run simulation for a given Adjustment class
def run_simulation(AdjustmentClass, T):
    adj = AdjustmentClass(trial=21, n_apps=n_apps, T=T, links=links,
                          max_n_states=max_n_states, max_n_flows=max_n_flows,
                          max_n_flow_hop=max_n_flow_hop, verbose=False)
    results = adj.run()

    # Collect metrics
    flex_values = [res["flex"] for res in results]
    time_values = [res["time"] for res in results]
    reconfig_values = [res["reconfig_count"] for res in results]

    avg_flex = sum(flex_values) / len(flex_values) if flex_values else 0
    avg_time = sum(time_values) / len(time_values) if time_values else 0
    reconfig_count_last = reconfig_values[-1] if reconfig_values else 0

    return {
        "avg_flex": avg_flex,
        "avg_time": avg_time,
        "reconfig_count_last": reconfig_count_last
    }

# Data storage for each method
results_heu = []
results_adjust1 = []
results_bipartite = []

# Run simulations across different T values with a progress bar
for T in tqdm(T_values, desc="Running simulations for different T values"):
    results_heu.append(run_simulation(Adjustment1, T))
    results_adjust1.append(run_simulation(Adjustment2, T))
    results_bipartite.append(run_simulation(Adjustment3, T))

# Plotting the results with markers and custom x-axis ticks
def plot_metric(metric, label):
    plt.figure()
    plt.plot(T_values, [res[metric] for res in results_heu], label="Heuristic Reconfiguration", marker='o')
    plt.plot(T_values, [res[metric] for res in results_adjust1], label="Guaranteed-Only Resource Partition", marker='o')
    plt.plot(T_values, [res[metric] for res in results_bipartite], label="Flexible Conditional Resource Partition", marker='o')
    plt.xlabel("Number of Slots (T)")
    plt.ylabel(label)
    plt.xticks(T_values)  # Set x-axis ticks to be exactly the T values
    plt.title(f"{label} vs. Number of Slots")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)  # Optional: add grid lines for readability
    plt.show()

# Generate the three required plots
plot_metric("avg_flex", "Average Flexibility")
plot_metric("avg_time", "Average Runtime")
plot_metric("reconfig_count_last", "Reconfiguration Count")

# Display results for review
print(results_heu)
print(results_adjust1)
print(results_bipartite)

'''
T_values = [30, 50, 70, 90, 110] 

results_heu = [{'avg_flex': 3.345598299999999, 'avg_time': 0.08871472263336182, 'reconfig_count_last': 142}, {'avg_flex': 3.695716960000005, 'avg_time': 0.11000938034057617, 'reconfig_count_last': 119}, {'avg_flex': 4.041337100000016, 'avg_time': 0.13917384862899781, 'reconfig_count_last': 106}, {'avg_flex': 3.694608859999998, 'avg_time': 0.20328735399246217, 'reconfig_count_last': 123}, {'avg_flex': 3.860507319999997, 'avg_time': 0.12720214939117433, 'reconfig_count_last': 74}]

results_adjust1 = [{'avg_flex': 3.3524262148026227, 'avg_time': 0.07486171245574952, 'reconfig_count_last': 125}, {'avg_flex': 3.6767903199278686, 'avg_time': 0.11435985088348388, 'reconfig_count_last': 119}, {'avg_flex': 4.041337100000016, 'avg_time': 0.13939221000671387, 'reconfig_count_last': 106}, {'avg_flex': 3.6865256546420446, 'avg_time': 0.2105297966003418, 'reconfig_count_last': 123}, {'avg_flex': 3.860507319999997, 'avg_time': 0.12843022966384887, 'reconfig_count_last': 74}]

results_bipartite = [{'avg_flex': 2.9048090269257414, 'avg_time': 0.03372742080688477, 'reconfig_count_last': 61}, {'avg_flex': 3.2416540028468943, 'avg_time': 0.08979939842224122, 'reconfig_count_last': 96}, {'avg_flex': 3.5043666909943294, 'avg_time': 0.07949325132369996, 'reconfig_count_last': 63}, {'avg_flex': 3.187374870870081, 'avg_time': 0.10873919582366944, 'reconfig_count_last': 78}, {'avg_flex': 2.9030073123877997, 'avg_time': 0.10016989517211915, 'reconfig_count_last': 59}]
'''