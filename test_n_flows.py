import importlib
import always_heu
import adjustment1trial
import bipartite_adjustment
import setcove_bipartite
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar

# Reload modules
importlib.reload(always_heu)
importlib.reload(adjustment1trial)
importlib.reload(bipartite_adjustment)
importlib.reload(setcove_bipartite)

from always_heu import Adjustment as Adjustment1
from adjustment1trial import Adjustment as Adjustment2
from bipartite_adjustment import Adjustment as Adjustment3
from setcove_bipartite import Adjustment as Adjustment4

# Simulation parameters
max_n_flows_values = [4,6, 8,10, 12,14, 16,18, 20]# Varying number of flows per state
n_apps = 5  
T = 50
links = range(30)
max_n_states = 8
max_n_flow_hop = 5

# Function to run simulation for a given Adjustment class
def run_simulation(AdjustmentClass, max_n_flows):
    adj = AdjustmentClass(trial=19, n_apps=n_apps, T=T, links=links,
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
results_setbip = []

# Run simulations across different max_n_flows values with a progress bar
for max_n_flows in tqdm(max_n_flows_values, desc="Running simulations for different max_n_flows"):
    
    results_heu.append(run_simulation(Adjustment1, max_n_flows))
    results_adjust1.append(run_simulation(Adjustment2, max_n_flows))
    '''
    results_bipartite.append(run_simulation(Adjustment3, max_n_flows))
    '''
    results_setbip.append(run_simulation(Adjustment4, max_n_flows))

# Plotting the results with markers and custom x-axis ticks
def plot_metric(metric, label):
    plt.figure()
    plt.plot(max_n_flows_values, [res[metric] for res in results_heu], label="Heuristic Reconfiguration", marker='o')
    plt.plot(max_n_flows_values, [res[metric] for res in results_adjust1], label="Guaranteed-Only Resource Partition", marker='o')
    #plt.plot(max_n_flows_values, [res[metric] for res in results_bipartite], label="Flexible Conditional Resource Partition", marker='o')
    plt.plot(max_n_flows_values, [res[metric] for res in results_setbip], label="SetCover+Bipartite", marker='o')
    plt.xlabel("Number of Flows per State (max_n_flows)")
    plt.ylabel(label)
    plt.xticks(max_n_flows_values)  # Set x-axis ticks to be exactly the max_n_flows values
    plt.title(f"{label} vs. Number of Flows per State")
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
print(results_setbip)
'''
max_n_flows_values = [4, 8, 12, 16, 20, 24] 

results_heu = [{'avg_flex': 5.0, 'avg_time': 0.3210597038269043, 'reconfig_count_last': 1}, {'avg_flex': 4.051887039999991, 'avg_time': 0.06295272874832153, 'reconfig_count_last': 98}, {'avg_flex': 3.2817993399999974, 'avg_time': 0.08725449800491333, 'reconfig_count_last': 129}, {'avg_flex': 3.110808559999996, 'avg_time': 0.11232919692993164, 'reconfig_count_last': 111}, {'avg_flex': 1.5413327000000039, 'avg_time': 0.17758500289916992, 'reconfig_count_last': 100}, {'avg_flex': 1.229961920000001, 'avg_time': 0.25261578512191774, 'reconfig_count_last': 53}]

results_adjust1 = [{'avg_flex': 5.0, 'avg_time': 0.2887871265411377, 'reconfig_count_last': 1}, {'avg_flex': 4.051887039999991, 'avg_time': 0.06589220237731934, 'reconfig_count_last': 98}, {'avg_flex': 3.2207932384670377, 'avg_time': 0.11153044939041137, 'reconfig_count_last': 144}, {'avg_flex': 3.124068421017243, 'avg_time': 0.10584564399719239, 'reconfig_count_last': 108}, {'avg_flex': 1.4538998173382143, 'avg_time': 0.15815520763397217, 'reconfig_count_last': 96}, {'avg_flex': 0.9537433752650533, 'avg_time': 0.19788089847564697, 'reconfig_count_last': 44}]

results_bipartite = [{'avg_flex': 5.0, 'avg_time': 0.2926969528198242, 'reconfig_count_last': 1}, {'avg_flex': 4.12308936151503, 'avg_time': 0.06736320543289184, 'reconfig_count_last': 95}, {'avg_flex': 2.4663232219078193, 'avg_time': 0.07313896417617798, 'reconfig_count_last': 91}, {'avg_flex': 2.403095128474849, 'avg_time': 0.06187588834762573, 'reconfig_count_last': 70}, {'avg_flex': 1.0891279049438425, 'avg_time': 0.07167350912094116, 'reconfig_count_last': 71}, {'avg_flex': 0.6878438626172263, 'avg_time': 0.08019232988357544, 'reconfig_count_last': 52}]

'''