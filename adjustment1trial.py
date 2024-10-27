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
        self.reconfig_count = 0
        self.infeasible_times = 0
        self.transition_times = 0
        self.time = 0

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
        k = 500
        results = []
        for _ in range(k):
            for i in range(self.n_apps):
                if len(self.infeasible_states[i]) > 0:
                    next_state = np.random.choice(
                        range(self.apps[i].n_states), p=self.apps[i].transitions[self.current_states[i]])
                    self.current_states[i] = next_state
                    self.transition_times += 1
                    if next_state in self.infeasible_states[i]:
                        if self.verbose:
                            print(f"App {i} enters an infeasible state")
                            print("current states:", self.current_states)
                        
                        self.infeasible_times += 1
                        inf_app = i
                        request = self.find_resource_request(inf_app, self.current_states[inf_app])
                        
                        if self.verbose:
                            print('request:', request)
                        #print('request:', request)
                        safe_provisions, unsafe_provisions = self.find_resource_provisions(inf_app, request)
                        if self.verbose:
                            print('safe provisions:', safe_provisions, 'unsafe provisions:', unsafe_provisions)
                            print('flexibility losses:', self.loss)
                        bt = time.time()
                        selected, total_loss, all_covered = self.solve_setcover(request, safe_provisions, self.loss)
                       
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
                            self.time += time.time()-bt
                            res = {"method": "adjustment", "n_affected_apps": len(
                                selected), "flex": self.flex, "time": time.time()-bt,  "reconfig_count":self.reconfig_count}
                        else:
                            if self.verbose:
                                print("unable to satisfy resource request")
                            bt = time.time()
                            self.heu2 = Heuristic(0, self.apps, self.links, self.T, self.current_states, verbose=False)
                            self.flex = self.heu2.run()
                            if self.flex == 0:
                                # if self.verbose:
                                print("reconfig also failed")
                                #return results
                            else:
                                self.partition = self.heu2.partition
                                self.feasible_states = self.heu2.all_feasible_states
                                self.infeasible_states = self.heu2.all_infeasible_states
                                self.reconfig_count += 1
                                res = {"method": "reconfig", "n_affected_apps": self.n_apps - 10,
                                       "flex": self.flex, "time": time.time()-bt,  "reconfig_count":self.reconfig_count}
                                self.time += time.time()-bt
                                print("reconfig", end=",")
                        print(self.flex)
                        results.append(res)
                    else:
                        print("still feasible")
                        results.append({"method": "adjustment", "n_affected_apps": 0,
                                       "flex": self.flex, "time": 0,  "reconfig_count":self.reconfig_count})
                    
                    if self.transition_times >= 500:
                        return results

        print(f"Number of reconfigurations: {self.reconfig_count}")
        return results



        

    def find_resource_request(self, i, s):
        # rough calculation, to be improved
        print('resource request start')
        flows = self.apps[i].states[s].flows

        resource_request = []
        cur_hops = {}
        for t in range(self.T):
            packets = []
            used_links = {}
            for f in flows:
                if t % f.period == 0:

                    if f.id in cur_hops and cur_hops[f.id] < len(f.txs):
                        '''
                        print('Current time: ',t, 'Flow period: ',f.period)
                        print('current hop of flow',f.id, 'is ',cur_hops[f.id], '. The len of links in flow is ', len(f.txs))
                        print('Flow links: ', f.txs)'''
                        
                        for i in range(cur_hops[f.id], len(f.txs)):
                            resource_request.append(
                            (f.txs[i], [(t//f.period)*f.period, (t//f.period+1)*f.period]))
                            #print('Append: ',f.txs[i])
                        
                           
 
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
                         max_n_states=20, max_n_flows=7, max_n_flow_hop=4,
                         verbose=False)
        results = adj.run()
        

     

        print(len(results), "\n\n\n")
        if len(results) >= 10:

            n_apps = [res["n_affected_apps"] for res in results]
            flexs = [res["flex"] for res in results]
            times = [res["time"] for res in results]
            counts = [res["reconfig_count"] for res in results]

            print({
                "n_adjusted_apps": n_apps,
                "flex": flexs,
                "time": times,
                "reconfig_count": counts,
                "xAxis": [i for i in range(len(flexs))],
            })
        print(adj.infeasible_times)
