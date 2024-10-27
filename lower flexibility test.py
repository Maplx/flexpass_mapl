import importlib
import always_heu
import adjustment1trial
import bipartite_adjustment
from tqdm import tqdm  # Import tqdm for progress bar
# Reload the modules to make sure changes are picked up
importlib.reload(always_heu)
importlib.reload(bipartite_adjustment)
from always_heu import Adjustment as Adjustment1
from bipartite_adjustment import Adjustment as Adjustment3

inf1 = []
total1 = []
all_results1 = {
    "n_adjusted_apps": [], "flex": [], "time": [], "reconfig_count": [], "xAxis": []}

inf2 = []
total2 = []
all_results2 = {
    "n_adjusted_apps": [], "flex": [], "time": [], "reconfig_count": [], "xAxis": []}


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Function to calculate percentage difference
def percentage_difference(value1, value2):
    return ((value2 - value1) / value1) * 100 if value1 != 0 else float('inf')

# Lists to store the results
topologies = []
flexibility_diffs = []
infeasible_state_freq_diffs = []

# Define the topologies you want to test (n_flows, n_flow_hop)
topology_parameters = [(7, 4), (7, 5), (7, 6), (8, 4), (8, 5), (8, 6), (9, 4), (9, 5), (9, 6)]

# Iterate through different topologies and collect data
with tqdm(total=len(topology_parameters), desc="Running Topologies") as pbar:
    for n_flows, n_flow_hop in topology_parameters:
        # Always Heuristic Adjustment
        adj1 = Adjustment1(trial=21, n_apps=10, T=50, links=range(30),
                           max_n_states=20, max_n_flows=n_flows, max_n_flow_hop=n_flow_hop,
                           verbose=False)
        results1 = adj1.run()
        
        inf1 = adj1.infeasible_times / len(results1)  # Frequency of infeasible state for heuristic reconfig

        # Bipartite Matching Adjustment
        adj3 = Adjustment3(trial=21, n_apps=10, T=50, links=range(30),
                           max_n_states=20, max_n_flows=n_flows, max_n_flow_hop=n_flow_hop,
                           verbose=False)
        results3 = adj3.run()
        inf3 = adj3.infeasible_times / len(results3)  # Frequency of infeasible state for bipartite matching

        # Calculate average flexibility
        avg_flex1 = np.mean([res["flex"] for res in results1])
        avg_flex3 = np.mean([res["flex"] for res in results3])

        # Calculate percentage differences
        flex_diff = percentage_difference(avg_flex3, avg_flex1)
        inf_state_diff = percentage_difference(inf3, inf1)

        # Store results
        topologies.append(f"Flows={n_flows}, Hops={n_flow_hop}")
        flexibility_diffs.append(flex_diff)
        infeasible_state_freq_diffs.append(inf_state_diff)

        # Update progress bar
        pbar.update(1)

# Plot the relationship between flexibility difference and infeasible state frequency difference
plt.figure(figsize=(10, 6))
plt.scatter(flexibility_diffs, infeasible_state_freq_diffs, color='blue')

# Label each point with its topology
for i, txt in enumerate(topologies):
    plt.annotate(txt, (flexibility_diffs[i], infeasible_state_freq_diffs[i]))

# Add axis labels and title
plt.xlabel('Flexibility Difference (%)')
plt.ylabel('Infeasible State Frequency Difference (%)')
plt.title('Relationship between Flexibility Difference and Infeasible State Frequency Difference')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
