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
links_values = [range(30), range(26), range(22), range(18), range(14), range(10)]  # Varying number of links
n_apps = 5  # Fixed number of applications (can adjust as needed)
T = 50
max_n_states = 10
max_n_flows = 8
max_n_flow_hop = 5

# Function to run simulation for a given Adjustment class
def run_simulation(AdjustmentClass, links):
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

# Run simulations across different links values with a progress bar
for links in tqdm(links_values, desc="Running simulations for different links ranges"):
    results_heu.append(run_simulation(Adjustment1, links))
    results_adjust1.append(run_simulation(Adjustment2, links))
    results_bipartite.append(run_simulation(Adjustment3, links))

# Plotting the results with markers and custom x-axis ticks
def plot_metric(metric, label):
    plt.figure()
    x_axis_values = [len(links) for links in links_values]  # Number of links in each range for x-axis
    plt.plot(x_axis_values, [res[metric] for res in results_heu], label="Heuristic Reconfiguration", marker='o')
    plt.plot(x_axis_values, [res[metric] for res in results_adjust1], label="Guaranteed-Only Resource Partition", marker='o')
    plt.plot(x_axis_values, [res[metric] for res in results_bipartite], label="Flexible Conditional Resource Partition", marker='o')
    plt.xlabel("Number of Links")
    plt.ylabel(label)
    plt.xticks(x_axis_values)  # Set x-axis ticks to be exactly the number of links
    plt.title(f"{label} vs. Number of Links")
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
links_values = [range(30), range(26), range(22), range(18), range(14), range(10)] 

results_heu = [{'avg_flex': 3.695716960000005, 'avg_time': 0.122424880027771, 'reconfig_count_last': 119}, {'avg_flex': 3.3476249200000048, 'avg_time': 0.10138954591751098, 'reconfig_count_last': 116}, {'avg_flex': 2.820051619999998, 'avg_time': 0.12184076642990112, 'reconfig_count_last': 155}, {'avg_flex': 2.1481846800000017, 'avg_time': 0.09111873865127564, 'reconfig_count_last': 121}, {'avg_flex': 1.8241032600000018, 'avg_time': 0.08521607875823975, 'reconfig_count_last': 123}, {'avg_flex': 1.2729582400000006, 'avg_time': 0.08082188558578492, 'reconfig_count_last': 163}]


results_adjust1 = [{'avg_flex': 3.6767903199278686, 'avg_time': 0.11971222925186158, 'reconfig_count_last': 119}, {'avg_flex': 3.3003099715230815, 'avg_time': 0.11614524459838867, 'reconfig_count_last': 144}, {'avg_flex': 2.4851253887888545, 'avg_time': 0.10390034484863281, 'reconfig_count_last': 128}, {'avg_flex': 2.207483666215782, 'avg_time': 0.07693501901626587, 'reconfig_count_last': 112}, {'avg_flex': 1.9374633948499163, 'avg_time': 0.06651782178878785, 'reconfig_count_last': 111}, {'avg_flex': 1.037787755089741, 'avg_time': 0.0694607229232788, 'reconfig_count_last': 119}]


results_bipartite = [{'avg_flex': 3.2416540028468943, 'avg_time': 0.09651825428009034, 'reconfig_count_last': 96}, {'avg_flex': 2.69941098519968, 'avg_time': 0.06884114646911621, 'reconfig_count_last': 80}, {'avg_flex': 1.83702025586711, 'avg_time': 0.045463492393493656, 'reconfig_count_last': 61}, {'avg_flex': 1.7257423666048473, 'avg_time': 0.029261953353881837, 'reconfig_count_last': 50}, {'avg_flex': 1.2851806090300568, 'avg_time': 0.0321789321899414, 'reconfig_count_last': 58}, {'avg_flex': 0.9618855685318615, 'avg_time': 0.019325562953948973, 'reconfig_count_last': 51}]


'''