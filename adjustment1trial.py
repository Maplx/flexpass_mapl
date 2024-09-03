import pulp
from typedefs import *
from app import App
from heuristic import Heuristic
import copy
import numpy as np
import random
import time


class Adjustment:
    def __init__(self, trial, n_apps, T, links, max_n_states, max_n_flows, max_n_flow_hop, verbose):
        self.n_apps = n_apps
        self.T = T
        self.links = links
        self.max_n_states = max_n_states
        self.max_n_flows = max_n_flows
        self.max_n_flow_hop = max_n_flow_hop
        self.verbose = verbose
        self.apps = [App(trial, i, self.links, self.T,
                         max_n_states=self.max_n_states,
                         max_n_flows=self.max_n_flows,
                         max_n_flow_hop=self.max_n_flow_hop)
                     for i in range(self.n_apps)]

    def run(self):
        self.current_states = [0]*self.n_apps
        self.heu = Heuristic(0, self.apps, self.links, self.T, self.current_states, verbose=False)
        flex = self.heu.run()
        self.flex = flex
        bt = time.time()
        if flex > 0 and flex < self.n_apps:
            print("init flex", flex)
            self.partition = self.heu.partition
            self.feasible_states = self.heu.all_feasible_states
            self.infeasible_states = self.heu.all_infeasible_states
            if self.verbose:
                print("infeasible states", self.infeasible_states)
            return self.transition()
        return []

    def transition(self):
        k = 2500
        results = []
        for _ in range(k):
            for i in range(self.n_apps):
                if len(self.infeasible_states[i]) > 0:
                    next_state = np.random.choice(
                        range(self.apps[i].n_states), p=self.apps[i].transitions[self.current_states[i]])
                    self.current_states[i] = next_state
                    if next_state in self.infeasible_states[i]:
                        if self.verbose:
                            print(f"App {i} enters an infeasible state")
                            print("current states:", self.current_states)

                        inf_app = i
                        request = self.find_resource_request(inf_app, self.current_states[inf_app])
                        if self.verbose:
                            print('request:', request)
                        safe_provisions, unsafe_provisions = self.find_resource_provisions(inf_app, request)
                        if self.verbose:
                            print('safe provisions:', safe_provisions, 'unsafe provisions:', unsafe_provisions)
                            print('flexibility losses:', self.loss)
                        bt = time.time()
                        selected, total_loss, all_covered = self.solve_setcover(request, safe_provisions, self.loss)
                        #all_covered = False
                        if all_covered:
                            if self.verbose:
                                print("Selected apps:", selected, "total loss:", total_loss)

                            self.flex = self.flex-total_loss
                            self.infeasible_states[inf_app].remove(self.current_states[inf_app])
                            self.feasible_states[inf_app].append(self.current_states[inf_app])
                            for a in selected:
                                self.feasible_states[a] = [s for s in self.feasible_states[a]
                                                           if s not in self.sacrificed_states[a]]
                                self.infeasible_states[a] = self.infeasible_states[a]+self.sacrificed_states[a]

                                for r in safe_provisions[a]:
                                    self.partition[r[1]][r[0]].app = inf_app
                                    self.partition[r[1]][r[0]].states = [self.current_states[inf_app]]
                            print(f"adjusted {len(selected)} apps", end=",")
                            res = {"method": "adjustment", "n_affected_apps": len(
                                selected), "flex": self.flex, "time": time.time()-bt}
                        else:
                            if self.verbose:
                                print("unable to satisfy resource request")
                            bt = time.time()
                            self.heu2 = Heuristic(0, self.apps, self.links, self.T, self.current_states, verbose=False)
                            self.flex = self.heu2.run()
                            if self.flex == 0:
                                # if self.verbose:
                                print("reconfig also failed")
                                return results
                            else:
                                self.partition = self.heu2.partition
                                self.feasible_states = self.heu2.all_feasible_states
                                self.infeasible_states = self.heu2.all_infeasible_states
                                res = {"method": "reconfig", "n_affected_apps": self.n_apps,
                                       "flex": self.flex, "time": time.time()-bt}
                                print("reconfig", end=",")
                        print(self.flex)
                        results.append(res)
                    else:
                        print("still feasible")
                        results.append({"method": "adjustment", "n_affected_apps": 0,
                                       "flex": self.flex, "time": 0})

        return results



        

    def find_resource_request(self, i, s):
        # rough calculation, to be improved
        flows = self.apps[i].states[s].flows

        resource_request = []
        cur_hops = {}
        for t in range(self.T):
            packets = []
            used_links = {}
            for f in flows:
                if t % f.period == 0:
                    if f.id in cur_hops and cur_hops[f.id] < len(f.txs):
                        resource_request.append(
                            (f.txs[cur_hops[f.id]], [(t//f.period)*f.period, (t//f.period+1)*f.period]))

                    cur_hops[f.id] = 0
                if cur_hops[f.id] < len(f.txs):
                    pkt = Packet(f.id, f.txs[cur_hops[f.id]], (t//f.period+1)*f.period, f.period)
                    packets.append(pkt)
            packets.sort(key=lambda x: x.deadline)

            for pkt in packets:
                if self.partition[t][pkt.link].app == i and pkt.link not in used_links:
                    used_links[pkt.link] = True
                    cur_hops[pkt.flow_id] += 1

        return resource_request


    def find_resource_provisions(self, inf_app, resource_request):
        safe_provisions = [[] for _ in range(self.n_apps)]
        unsafe_provisions = [[] for _ in range(self.n_apps)]

        self.sacrificed_states = [[] for _ in range(self.n_apps)]
        for r in resource_request:
            for t in [r[1][0], r[1][1]-1]:
                i = self.partition[t][r[0]].app
                if i != -1 and i != inf_app:
                    if self.current_states[i] not in self.partition[t][r[0]].states:
                        safe_provisions[i].append((r[0], t))
                        self.sacrificed_states[i] += self.partition[t][r[0]].states
                    else:
                        unsafe_provisions[i].append((r[0], t))

        self.loss = [0]*self.n_apps

        for i in range(self.n_apps):
            old_flex = self.calculate_flexibility(i, self.feasible_states[i], self.infeasible_states[i])
            self.sacrificed_states[i] = list(set(self.sacrificed_states[i]))
            new_flex = self.calculate_flexibility(
                i, [s for s in self.feasible_states[i] if s not in self.sacrificed_states[i]], self.infeasible_states[i]+self.sacrificed_states[i])
            self.loss[i] = old_flex-new_flex

        return safe_provisions, unsafe_provisions


    def calculate_flexibility(self, i, feasible_states, infeasible_states):
        gamma = 0.9

        M_prime = copy.deepcopy(self.apps[i].transitions)
        for s in range(len(M_prime)):
            if s in infeasible_states:
                M_prime[s, :] = 0
                M_prime[s, s] = 1
        flex = 0
        denominator = 0
        for k in range(1, self.apps[i].k_max+1):
            k_step_matrix = np.linalg.matrix_power(M_prime, k)
            k_step_success_prob = sum(k_step_matrix[0, :][s] for s in feasible_states)
            flex += (gamma**k)*k_step_success_prob
            denominator += (gamma**k)
        flex = flex/denominator
        return flex

    def solve_setcover(self, RR, RP, loss):
        # Create the LP problem
        prob = pulp.LpProblem("Weighted_Set_Cover", pulp.LpMinimize)

        # Create binary variables for each subset in RP
        x = {i: pulp.LpVariable(f"x{i}", cat='Binary') for i in range(len(RP))}

        # Objective function: minimize the total weight
        prob += 5*pulp.lpSum(x[i] for i in range(len(RP))) + pulp.lpSum(loss[i] * x[i] for i in range(len(RP)))

        # Constraints: each resource request must be covered at least once
        for e, T in RR:
            prob += pulp.lpSum(x[i] for i in range(len(RP))
                               if any((e, t) in RP[i] for t in range(T[0], T[1] + 1))) >= 1

        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # Extract the solution
        selected_subsets = [i for i in range(len(RP)) if x[i].value() == 1]
        total_loss = sum(loss[i] for i in selected_subsets)
        all_covered = all(
            any(any((e, t) in RP[i] for t in range(T[0], T[1] + 1)) for i in selected_subsets)
            for e, T in RR
        )
        return selected_subsets, total_loss, all_covered
    
    



if __name__ == "__main__":
    cnt_no_need = 0
    cnt_adj_success = 0
    cnt_adj_fail = 0
    avg_flex_loss = 0
    avg_affected_apps = 0
    n_trials = 1
    for i in range(n_trials):
        adj = Adjustment(trial=21, n_apps=10, T=50, links=range(30),
                         max_n_states=20, max_n_flows=8, max_n_flow_hop=5,
                         verbose=False)
        results = adj.run()

     

        print(len(results), "\n\n\n")
        if len(results) >= 50:

            n_apps = [res["n_affected_apps"] for res in results]
            flexs = [res["flex"] for res in results]
            times = [res["time"] for res in results]

            print({
                "n_adjusted_apps": n_apps,
                "flex": flexs,
                "time": times,
                "xAxis": [i for i in range(len(flexs))],
            })

'''
init flex 2.5706
adjusted 3 apps,2.295925327803344
adjusted 3 apps,1.9959287700463477
adjusted 3 apps,1.9214909672787521
adjusted 4 apps,1.4755698809382793
adjusted 3 apps,1.3277650760399644
adjusted 1 apps,1.3277650760399644
reconfig,2.8752
reconfig,2.90555
adjusted 4 apps,2.90555
adjusted 3 apps,2.839656879319074
adjusted 4 apps,2.4744118545105875
reconfig,3.10479
reconfig,2.71308
adjusted 3 apps,2.71308
adjusted 1 apps,2.71308
reconfig,3.00223
adjusted 4 apps,2.9527526551687036
adjusted 2 apps,2.9527526551687036
adjusted 3 apps,2.9527526551687036
adjusted 4 apps,2.9527526551687036
adjusted 2 apps,2.9527526551687036
adjusted 3 apps,2.8278935530870846
adjusted 2 apps,2.8278935530870846
adjusted 5 apps,2.768043076176051
adjusted 3 apps,2.768043076176051
adjusted 2 apps,2.768043076176051
adjusted 2 apps,2.768043076176051
adjusted 2 apps,2.768043076176051
reconfig,3.03148
reconfig,2.85434
adjusted 5 apps,2.7660459472753756
adjusted 3 apps,2.7660459472753756
reconfig,3.07975
adjusted 2 apps,3.0411797302624928
adjusted 4 apps,3.0411797302624928
adjusted 1 apps,3.0411797302624928
adjusted 0 apps,3.0411797302624928
reconfig,3.27522
adjusted 1 apps,3.27522
reconfig,3.27138
reconfig,3.39516
reconfig,3.47427
adjusted 3 apps,3.431979155820591
adjusted 4 apps,3.4053357687209043
adjusted 4 apps,3.08935187743269
reconfig,3.28029
adjusted 8 apps,2.2260842346377725
reconfig,3.38308
adjusted 3 apps,3.38308
adjusted 2 apps,3.31727993899702
adjusted 2 apps,3.280484835556222
reconfig,3.40644
adjusted 2 apps,3.40644
adjusted 1 apps,3.40644
reconfig,2.59245
adjusted 3 apps,2.59245
reconfig,3.43124
adjusted 3 apps,2.7041485596735226
adjusted 3 apps,2.7041485596735226
'''
