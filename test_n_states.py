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
max_n_states_values = [4,5,6,7, 8,9, 10,11, 12,13, 14,15] # Varying number of states per application
n_apps = 5  
T = 50
links = range(30)
max_n_flows = 8
max_n_flow_hop = 5

# Function to run simulation for a given Adjustment class
def run_simulation(AdjustmentClass, max_n_states):
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

# Run simulations across different max_n_states values with a progress bar
for max_n_states in tqdm(max_n_states_values, desc="Running simulations for different max_n_states"):
    results_heu.append(run_simulation(Adjustment1, max_n_states))
    results_adjust1.append(run_simulation(Adjustment2, max_n_states))
    '''
    results_bipartite.append(run_simulation(Adjustment3, max_n_states))
    '''
    results_setbip.append(run_simulation(Adjustment4, max_n_states))

# Plotting the results with markers and custom x-axis ticks
def plot_metric(metric, label):
    plt.figure()
    plt.plot(max_n_states_values, [res[metric] for res in results_heu], label="Heuristic Reconfiguration", marker='o')
    plt.plot(max_n_states_values, [res[metric] for res in results_adjust1], label="Guaranteed-Only Resource Partition", marker='o')
    #plt.plot(max_n_states_values, [res[metric] for res in results_bipartite], label="Flexible Conditional Resource Partition", marker='o')
    plt.plot(max_n_states_values, [res[metric] for res in results_setbip], label="SetCover+Bipartite", marker='o')
    plt.xlabel("Number of States per Application (max_n_states)")
    plt.ylabel(label)
    plt.xticks(max_n_states_values)  # Set x-axis ticks to be exactly the max_n_states values
    plt.title(f"{label} vs. Number of States per Application")
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
max_n_states_values = [4, 6, 8, 10, 12, 14, 16, 18]

results_heu = [{'avg_flex': 4.177818000000001, 'avg_time': 0.025963497161865235, 'reconfig_count_last': 2}, {'avg_flex': 3.989331479999989, 'avg_time': 0.06371417427062988, 'reconfig_count_last': 103},{'avg_flex': 4.051887039999991, 'avg_time': 0.06253665828704834, 'reconfig_count_last': 98}, {'avg_flex': 3.695716960000005, 'avg_time': 0.10873607015609742, 'reconfig_count_last': 119}, {'avg_flex': 3.092873559999999, 'avg_time': 0.1524033088684082, 'reconfig_count_last': 132}, {'avg_flex': 3.078731040000001, 'avg_time': 0.11879457330703735, 'reconfig_count_last': 100}, {'avg_flex': 3.0494044400000004, 'avg_time': 0.13470645475387574, 'reconfig_count_last': 102}, {'avg_flex': 3.104171039999997, 'avg_time': 0.19129321384429931, 'reconfig_count_last': 133}]

results_adjust1= [{'avg_flex': 4.177818000000001, 'avg_time': 0.026328452428181968, 'reconfig_count_last': 2},{'avg_flex': 3.9832228908864775, 'avg_time': 0.06065213441848755, 'reconfig_count_last': 100}, {'avg_flex': 4.051887039999991, 'avg_time': 0.061348286151885985, 'reconfig_count_last': 98}, {'avg_flex': 3.6767903199278686, 'avg_time': 0.10854367876052856, 'reconfig_count_last': 119}, {'avg_flex': 3.0086288515483806, 'avg_time': 0.14093067502975465, 'reconfig_count_last': 122}, {'avg_flex': 3.073758759962886, 'avg_time': 0.11558521032333374, 'reconfig_count_last': 91}, {'avg_flex': 2.742787788058838, 'avg_time': 0.1281854419708252, 'reconfig_count_last': 95}, {'avg_flex': 3.0229559655952714, 'avg_time': 0.17858758401870727, 'reconfig_count_last': 128}]

results_bipartite = [{'avg_flex': 3.92, 'avg_time': 0.027655808130900066, 'reconfig_count_last': 2}, {'avg_flex': 3.8968337322761375, 'avg_time': 0.057703036308288574, 'reconfig_count_last': 98}, {'avg_flex': 4.12308936151503, 'avg_time': 0.05908630084991455, 'reconfig_count_last': 95}, {'avg_flex': 3.2416540028468943, 'avg_time': 0.09093670845031739, 'reconfig_count_last': 96}, {'avg_flex': 2.4073667452155103, 'avg_time': 0.08370793962478637, 'reconfig_count_last': 74}, {'avg_flex': 2.355802782226752, 'avg_time': 0.07532717990875244, 'reconfig_count_last': 60}, {'avg_flex': 2.30308986784307, 'avg_time': 0.08219743251800538, 'reconfig_count_last': 62}, {'avg_flex': 2.230057580948673, 'avg_time': 0.08003264141082764, 'reconfig_count_last': 55}]

'''