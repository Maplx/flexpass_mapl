import importlib
import always_heu
import adjustment1trial
import bipartite_adjustment

# Reload the modules to make sure changes are picked up
importlib.reload(always_heu)
importlib.reload(adjustment1trial)
importlib.reload(bipartite_adjustment)

from always_heu import Adjustment as Adjustment1
from adjustment1trial import Adjustment as Adjustment2
from bipartite_adjustment import Adjustment as Adjustment3

inf1 = []
total1 = []
all_results1 = {
    "n_adjusted_apps": [], "flex": [], "time": [], "reconfig_count": [], "xAxis": []}

inf2 = []
total2 = []
all_results2 = {
    "n_adjusted_apps": [], "flex": [], "time": [], "reconfig_count": [], "xAxis": []}

inf3 = []
total3 = []
all_results3 = {
    "n_adjusted_apps": [], "flex": [], "time": [], "reconfig_count": [], "xAxis": []}

for n in [7]:
    for h in [4]:
        # Always Heuristic
        adj = Adjustment1(trial=21, n_apps=10, T=50, links=range(30),
                          max_n_states=20, max_n_flows=n, max_n_flow_hop=h,
                          verbose=False)
        results = adj.run()
        inf1.append(adj.infeasible_times)
        total1.append(len(results))

        # Extract data from results
        n_apps = [res["n_affected_apps"] for res in results]
        flexs = [res["flex"] for res in results]
        times = [res["time"] for res in results]
        counts = [res["reconfig_count"] for res in results]

        # Append to the overall results dictionary
        all_results1["n_adjusted_apps"].extend(n_apps)
        all_results1["flex"].extend(flexs)
        all_results1["time"].extend(times)
        all_results1["reconfig_count"].extend(counts)
        all_results1["xAxis"].extend([i for i in range(len(flexs))])

        # Set Cover (Adjustment1trial)
        adj = Adjustment2(trial=21, n_apps=10, T=50, links=range(30),
                          max_n_states=20, max_n_flows=n, max_n_flow_hop=h,
                          verbose=False)
        results = adj.run()
        inf2.append(adj.infeasible_times)
        total2.append(len(results))

        # Extract data from results
        n_apps = [res["n_affected_apps"] for res in results]
        flexs = [res["flex"] for res in results]
        times = [res["time"] for res in results]
        counts = [res["reconfig_count"] for res in results]

        # Append to the overall results dictionary
        all_results2["n_adjusted_apps"].extend(n_apps)
        all_results2["flex"].extend(flexs)
        all_results2["time"].extend(times)
        all_results2["reconfig_count"].extend(counts)
        all_results2["xAxis"].extend([i for i in range(len(flexs))])

        # Bipartite Adjustment
        adj = Adjustment3(trial=21, n_apps=10, T=50, links=range(30),
                          max_n_states=20, max_n_flows=n, max_n_flow_hop=h,
                          verbose=False)
        results = adj.run()
        inf3.append(adj.infeasible_times)
        total3.append(len(results))

        # Extract data from results
        n_apps = [res["n_affected_apps"] for res in results]
        flexs = [res["flex"] for res in results]
        times = [res["time"] for res in results]
        counts = [res["reconfig_count"] for res in results]

        # Append to the overall results dictionary
        all_results3["n_adjusted_apps"].extend(n_apps)
        all_results3["flex"].extend(flexs)
        all_results3["time"].extend(times)
        all_results3["reconfig_count"].extend(counts)
        all_results3["xAxis"].extend([i for i in range(len(flexs))])

print('always heu:', inf1, total1)
print('always heu freq:', sum(inf1) / sum(total1))
print('set cover:', inf2, total2)
print('set cover freq:', sum(inf2) / sum(total2))
print('bipartite:', inf3, total3)
print('bipartite freq:', sum(inf3) / sum(total3))
