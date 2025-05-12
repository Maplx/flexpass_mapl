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
n_apps_values = [4, 5, 6, 7, 8, 9, 10, 11, 12] # Varying number of applications
T = 50
links = range(30)
max_n_states = 10
max_n_flows = 8
max_n_flow_hop = 5

# Function to run simulation for a given Adjustment class
def run_simulation(AdjustmentClass, n_apps):
    adj = AdjustmentClass(trial=25, n_apps=n_apps, T=T, links=links,
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
# Run simulations across different n_apps values with a progress bar
for n_apps in tqdm(n_apps_values, desc="Running simulations for different n_apps"):

    results_heu.append(run_simulation(Adjustment1, n_apps))
    results_adjust1.append(run_simulation(Adjustment2, n_apps))
    '''
    results_bipartite.append(run_simulation(Adjustment3, n_apps))
    '''
    results_setbip.append(run_simulation(Adjustment4, n_apps))


# Plotting the results with markers and custom x-axis ticks
def plot_metric(metric, label):
    plt.figure()
    plt.plot(n_apps_values, [res[metric] for res in results_heu], label="Heuristic Reconfiguration", marker='o')
    plt.plot(n_apps_values, [res[metric] for res in results_adjust1], label="Guaranteed-Only Resource Partition ", marker='o')
    #plt.plot(n_apps_values, [res[metric] for res in results_bipartite], label="Flexible Conditional Resource Partition", marker='o')
    plt.plot(n_apps_values, [res[metric] for res in results_setbip], label="SetCover+Bipartite", marker='o')
    plt.xlabel("Number of Applications (n_apps)")
    plt.ylabel(label)
    plt.xticks(n_apps_values)  # Set x-axis ticks to be exactly the n_apps values
    plt.title(f"{label} vs. Number of Applications")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)  # Optional: add grid lines for readability
    plt.show()

# Generate the three required plots

plot_metric("avg_flex", "Average Flexibility")
plot_metric("avg_time", "Average Runtime")
plot_metric("reconfig_count_last", " Reconfiguration Count")



print(results_heu)
print(results_adjust1)
print(results_bipartite)
print(results_setbip)

'''
n_apps_values = [4, 6, 8, 10, 12]
results_heu = [{'avg_flex': 3.7302443749999976, 'avg_time': 0.05717889219522476, 'reconfig_count_last': 6}, {'avg_flex': 3.92, 'avg_time': 0.16960632514953613, 'reconfig_count_last': 61}, {'avg_flex': 4.187214399999994, 'avg_time': 0.27583341789245606, 'reconfig_count_last': 126}, {'avg_flex': 3.8480216399999954, 'avg_time': 0.3135916690826416, 'reconfig_count_last': 117}, {'avg_flex': 2.956260120000008, 'avg_time': 0.5454160766601562, 'reconfig_count_last': 94}]

results_adjust1 = [{'avg_flex': 3.7302443749999976, 'avg_time': 0.05514984577894211, 'reconfig_count_last': 6}, {'avg_flex': 3.61, 'avg_time': 0.10583919143676758, 'reconfig_count_last': 54}, {'avg_flex': 3.986485708271972, 'avg_time': 0.22948332643508912, 'reconfig_count_last': 121}, {'avg_flex': 3.7523132704268662, 'avg_time': 0.2791166958808899, 'reconfig_count_last': 106}, {'avg_flex': 1.9539163583426573, 'avg_time': 0.2314671401977539, 'reconfig_count_last': 78}]

results_bipartite = [{'avg_flex': 3.7302443749999976, 'avg_time': 0.053862065076828, 'reconfig_count_last': 6}, {'avg_flex': 3.5970371744367927, 'avg_time': 0.06002099990844727, 'reconfig_count_last': 44}, {'avg_flex': 3.300099782022061, 'avg_time': 0.1172977032661438, 'reconfig_count_last': 73}, {'avg_flex': 2.9417762353429633, 'avg_time': 0.12196734476089477, 'reconfig_count_last': 53}, {'avg_flex': 1.7416209276802306, 'avg_time': 0.11321051502227783, 'reconfig_count_last': 45}]

'''