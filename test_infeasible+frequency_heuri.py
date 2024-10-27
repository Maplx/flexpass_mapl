from always_heu import Adjustment

l = []
w = []

for t in [30,50,70]:
    for i in [7,8,9]:
        for j in [3,4,5]:
            adj = Adjustment(trial=21, n_apps=10, T=t, links=range(30),
                         max_n_states=20, max_n_flows=i, max_n_flow_hop=j,
                         verbose=False)
            results = adj.run()
            
            l.append(adj.infeasible_times)
            w.append(len(results))

print(l,w)