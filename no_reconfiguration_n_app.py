import pulp
from typedefs import *
from app import App
import importlib
import heuristic
importlib.reload(heuristic)
import heuristic_2
from heuristic_2 import Hueristic_2
import matplotlib.pyplot as plt
from heuristic import Heuristic
import copy
import numpy as np
import random
import time
from tqdm import tqdm



class Adjustment1:
    def __init__(self, trial, n_apps, T, links, max_n_states, max_n_flows, max_n_flow_hop, verbose):
        
        self.n_apps = n_apps
        self.T = T
        self.links = links
        self.max_n_states = max_n_states
        self.max_n_flows = max_n_flows
        self.max_n_flow_hop = max_n_flow_hop
        self.verbose = verbose
        self.trial = trial
        random.seed(self.trial)
        self.apps = [App(trial, i, self.links, self.T,
                         max_n_states=self.max_n_states,
                         max_n_flows=self.max_n_flows,
                         max_n_flow_hop=self.max_n_flow_hop)
                     for i in range(self.n_apps)]
        self.reconfig_count = 0
        self.infeasible_times = 0
        self.transition_times = 0
        self.time = 0
        self.n_adjust = 0

    def run(self):
        random.seed(self.trial)
        self.current_states = [random.randint(0, len(self.apps[_].states)-1) for _ in range(self.n_apps)]
        print(self.current_states)
        
        self.heu = Heuristic(0, self.apps, self.links, self.T,self.current_states, verbose=False)
        flex = self.heu.run()
        self.flex = flex
        print('flex:',flex)
        bt = time.time()
        if flex > 0 and flex <= self.n_apps:
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
                        print(f"App {i} enters an infeasible state")
                        self.infeasible_times += 1
                        inf_app = i
                        
                        bt = time.time()
                        all_covered = False
                        if all_covered:
                            all_covered = False
                        else:
                           
                            bt = time.time()
                            self.heu2 = Heuristic(0, self.apps, self.links, self.T, self.current_states, verbose=False)
                            self.flex = self.heu2.run()
                            
                            if self.flex == 0:
                                # if self.verbose:
                                print("reconfig also failed")
                                res = {"method": "reconfig", "n_affected_apps": 0,
                                       "flex": 0, "time": time.time()-bt,  "reconfig_count":self.reconfig_count}
                                results.append(res)

                                return results

                            else:
                                self.partition = self.heu2.partition
                                self.feasible_states = self.heu2.all_feasible_states
                                self.infeasible_states = self.heu2.all_infeasible_states
                                self.reconfig_count += 1
                                self.time += time.time()-bt
                                print('reconfig: ',self.flex)
                                res = {"method": "reconfig", "n_affected_apps": self.n_apps,
                                       "flex": self.flex, "time": time.time()-bt,  "reconfig_count":self.reconfig_count}
                                self.n_adjust = self.n_apps
                                results.append(res)
                                return results
                        
                    else:
                        print("still feasible")
                else:
                    res = {"method": "reconfig", "n_affected_apps": 0,
                                       "flex": self.flex, "time": 0,  "reconfig_count":self.reconfig_count}
                    results.append(res)

        print(f"Number of reconfigurations: {self.reconfig_count}")
        print('full flex')

        return results

#####################################################################################################################################
    
class Adjustment2:
    def __init__(self, trial, n_apps, T, links, max_n_states, max_n_flows, max_n_flow_hop, verbose):
        self.n_apps = n_apps
        self.T = T
        self.links = links
        self.max_n_states = max_n_states
        self.max_n_flows = max_n_flows
        self.max_n_flow_hop = max_n_flow_hop
        self.verbose = verbose
        self.trial = trial
        random.seed(self.trial)
        self.apps = [App(trial, i, self.links, self.T,
                         max_n_states=self.max_n_states,
                         max_n_flows=self.max_n_flows,
                         max_n_flow_hop=self.max_n_flow_hop)
                     for i in range(self.n_apps)]
        self.reconfig_count = 0
        self.infeasible_times = 0
        self.transition_times = 0
        self.time = 0
        self.n_adjust = 0

    def run(self):
        random.seed(self.trial)
        self.current_states = [random.randint(0, len(self.apps[_].states)-1) for _ in range(self.n_apps)]
        print('current_states:', self.current_states)

        self.heu = Heuristic(0, self.apps, self.links, self.T, self.current_states, verbose=False)
        flex = self.heu.run()
        self.flex = flex
        bt = time.time()
        print('flex:',flex)

        if flex > 0 and flex <= self.n_apps:
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
             
                        print(f"App {i} enters an infeasible state")
                        
              
                        
                        self.infeasible_times += 1
                        inf_app = i
                        request = self.find_resource_request(inf_app, self.current_states[inf_app])

                        print('previous req len:', len(request))

                        for r in request:
                            for t in [r[1][0], r[1][1]-1]:
                                a = self.partition[t][r[0]].app
                                if a == -1:
                                    self.partition[t][r[0]].app = inf_app
                                    if r in request:
                                        request.remove(r)
                        
                        print('after idle len:', len(request))
     
 
                        #print('request:', request)
                        safe_provisions, unsafe_provisions = self.find_resource_provisions(inf_app, request)
                    
                
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
                            if self.flex <= 0:
                                self.flex = self.heu.calculate_flexibility()
                            res = {"method": "adjustment", "n_affected_apps": len(
                                selected), "flex": self.flex, "time": time.time()-bt,  "reconfig_count":self.reconfig_count}
                            results.append(res)
                            self.n_adjust = len(selected)
                            print('set cover worked')
                            return results
                        
                        else:
                            self.flex = 0
                            print('set cover failed')
                            res = {"method": "reconfig", "n_affected_apps": 0,
                                       "flex": 0, "time": time.time()-bt,  "reconfig_count":self.reconfig_count}
                            results.append(res)
                            return results
                        
                    else:
                        print("still feasible")
                else:
                    res = {"method": "reconfig", "n_affected_apps": 0,
                                       "flex": self.flex, "time": 0,  "reconfig_count":self.reconfig_count}
                    results.append(res)

        print('nothing happens')
                    
        print('full flex')

        return results
    
    def find_resource_request(self, i, s):
        #rough calculation, to be improved
        #print('resource request start')
        
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

            

    
    def find_provision(self, inf_app, single_link_resource_request):
        provision = [[] for _ in range(self.n_apps +1)]
        self.sacrificed_states = [[] for _ in range(self.n_apps+1)]
        self.condi_states = [[] for _ in range(self.n_apps+1)]

        matching_request = [[] for _ in range(self.n_apps+1)]
        matching_request[inf_app] = single_link_resource_request
        #print(single_link_resource_request)

        

        for r in single_link_resource_request:

            for t in [r[1][0], r[1][1]-1]:
                i = self.partition[t][r[0]].app
                if i != -1 and i != inf_app:
                    provision[i].append((r[0],t))

                    #rint(i)

                    if self.current_states[i] not in self.partition[t][r[0]].states:
                        provision[i].remove((r[0],t))
                        
        
                        #self.sacrificed_states[i] += self.partition[t][r[0]].states
                    else:
                        #print('condi:',i)
                        s = self.current_states[i]
                        flow = self.apps[i].states[s].flows
                        a = t//flow[0].period
                        for f in flow:
                            if r[0] in f.txs:
                                a = t//f.period
                        matching_request[i].append((r[0],[a*f.period, (a+1)*f.period]))
                        self.condi_states[i] += self.partition[t][r[0]].states
                
                if i == -1:
                    provision[i].append((r[0],t))

        self.loss = [0]*self.n_apps
        
        for i in range(self.n_apps):
            old_flex = self.calculate_flexibility(i, self.feasible_states[i], self.infeasible_states[i])
            self.sacrificed_states[i] = list(set(self.sacrificed_states[i]))
            new_flex = self.calculate_flexibility(
                i, [s for s in self.feasible_states[i] if s not in self.sacrificed_states[i]]
                , self.infeasible_states[i]+self.sacrificed_states[i])
            self.loss[i] = old_flex-new_flex
        '''
        print(provision)
        print(self.loss)
        print(matching_request)
        '''
        return provision, matching_request
    
    def hungarian_matching(self, provision, loss, matching_request):
        # Prepare a list of valid requests and provisions
        valid_requests = []
        valid_provisions = []
        request_to_app = []
        provision_to_app = []

        # Collect all valid requests
        for req_app in range(len(matching_request)):
            for req in matching_request[req_app]:
                valid_requests.append(req)
                request_to_app.append(req_app)

        # Collect all valid provisions
        for prov_app in range(len(provision)):
            if provision[prov_app]:  # Only consider apps with provisions
                valid_provisions.append(provision[prov_app])
                provision_to_app.append(prov_app)

        # If there are no valid requests or provisions, return no matching
        if not valid_requests or not valid_provisions:
            return None, float('inf')  # Indicate no matching possible

        n_requests = len(valid_requests)
        n_provisions = len(valid_provisions)
        cost_matrix = np.full((n_requests, n_provisions), 9999.0)  # Large value for invalid matches
        

        # Populate the cost matrix
        for req_idx, (req_link, req_time_range) in enumerate(valid_requests):
            for prov_idx, prov_links in enumerate(valid_provisions):
                for prov_link, prov_time in prov_links:
                    if prov_link == req_link and req_time_range[0] <= prov_time <= req_time_range[1]:
                        # Set the cost as the loss of the provider
                        cost_matrix[req_idx][prov_idx] = 0
        
        print('cost matrix:',cost_matrix)

        # Apply the Hungarian algorithm to the valid part of the cost matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Collect the matching result
        matching = []
        total_cost = 0
        for row, col in zip(row_ind, col_ind):
            if cost_matrix[row, col] < 9999:  # Only valid matches
                matching.append((request_to_app[row], provision_to_app[col]))
                total_cost += cost_matrix[row, col]
            else:
                return None, float('inf')  # Indicate that a valid matching was not found

        return matching, total_cost


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




#######################################################################################################################
import pulp
from typedefs import *
from app import App
import importlib

import heuristic
importlib.reload(heuristic)

from heuristic import Heuristic
import copy
import numpy as np
import random
import time
from itertools import groupby
from operator import itemgetter
from scipy.optimize import linear_sum_assignment
import numpy as np


class Adjustment3:
    def __init__(self, trial, n_apps, T, links, max_n_states, max_n_flows, max_n_flow_hop, verbose):
        
        self.n_apps = n_apps
        self.T = T
        self.links = links
        self.max_n_states = max_n_states
        self.max_n_flows = max_n_flows
        self.max_n_flow_hop = max_n_flow_hop
        self.verbose = verbose
        self.trial = trial
        random.seed(self.trial)
        self.apps = [App(trial, i, self.links, self.T,
                         max_n_states=self.max_n_states,
                         max_n_flows=self.max_n_flows,
                         max_n_flow_hop=self.max_n_flow_hop)
                     for i in range(self.n_apps)]
        self.reconfig_count = 0
        self.infeasible_times = 0
        self.transition_times = 0
        self.time = 0
        self.n_adjust = 0

    def run(self):
        random.seed(self.trial)
        self.current_states = [random.randint(0, len(self.apps[_].states)-1) for _ in range(self.n_apps)]
        self.heu = Heuristic(0, self.apps, self.links, self.T, self.current_states, verbose=False)
        flex = self.heu.run()
        self.flex = flex
        bt = time.time()
        print('flex:',flex)
        if flex > 0 and flex <= self.n_apps:
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

                        
                        for r in request:
                            for t in [r[1][0], r[1][1]-1]:
                                a = self.partition[t][r[0]].app
                                if a == -1:
                                    self.partition[t][r[0]].app = inf_app
                                    if r in request:
                                        request.remove(r)
                        
                        safe_provisions, unsafe_provisions = self.find_resource_provisions(inf_app, request)
              
                        bt = time.time()
                        t_set = 0
  
                        selected, total_loss, all_covered = self.solve_setcover(request, safe_provisions, self.loss)

                        if all_covered:
                            print('all covered')
                           
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
                            t_set += time.time()-bt
                            self.time += t_set
                            res = {"method": "adjustment", "n_affected_apps": len(
                                selected), "flex": self.flex, "time": time.time()-bt,  "reconfig_count":self.reconfig_count}
                            results.append(res)
                            self.n_adjust = len(selected)
                            return results
 
              
                        if not all_covered:
                            print('not all covered')
                            sel = 0
                            for p in range(len(safe_provisions)):
                                if len(safe_provisions[p]) > 0:
                                    sel += 1
                                    for ls in safe_provisions[p]:
                                        l = ls[0]
                                        s = ls[1]
                                        self.partition[s][l].app = inf_app
                                        for r in request:
                                            if l == r[0] and r[1][0] <= s <=r[1][1]:
                                                request.remove(r)
                                                break            
            
                            ac = True
                            request_sorted = sorted(request, key=itemgetter(0))
                            grouped_resource = [list(group) for key, group in groupby(request_sorted, key=itemgetter(0))]

                            total_loss = 0
                        
                            n_app = 0
                            matched_apps = []


                            for l in grouped_resource:
                                print(l)
                                provision, matching_request = self.find_provision(inf_app, single_link_resource_request=l)
                                matching, total_cost = self.hungarian_matching(provision,self.loss, matching_request)
                                total_loss += total_cost
                                #print(matching, total_cost)
                                if matching is None:
                                    ac = False
                                    break
                            
                                for m in matching:
                                    if m[1] not in matched_apps and m[1] != -1:
                                        matched_apps.append(m[1])
                                        n_app += 1

                                    a = m[1]
                                    
                                
                                    for r in provision[a]:
                                        self.partition[r[1]][r[0]].app = inf_app
                                        self.partition[r[1]][r[0]].states = [self.current_states[inf_app]]

                                    self.heu.calculate_flexibility()
                    
       
                            if ac:
                               
                                print( "total loss:", total_loss)
                                self.heu.calculate_flexibility()
                           

                                self.flex = self.flex-total_loss
                       
                                

                                print(n_app, 'apps engaged in matching adjustment')
                                self.time += time.time()-bt
                                self.time += t_set
                                print('Flexibility after matching:', self.flex)
                                res = {"method": "match","n_affected_apps": n_app+sel, "flex": self.flex, "time": time.time()-bt,  "reconfig_count":self.reconfig_count}
                                results.append(res)
                                print('set cover bipartite worked')
                                self.n_adjust = n_app+sel
                                return results
                            
                            if ac == False:
                                self.flex = 0
                                print('set cover bipartite failed')
                                res = {"method": "match","n_affected_apps": 0, "flex": 0, "time": time.time()-bt,  "reconfig_count":self.reconfig_count}
                                results.append(res)
                                return results
                               
                    else:
                        print("still feasible")
                else:
                    print('nothing happened')
                    res = {"method": "reconfig", "n_affected_apps": 0,
                                       "flex": self.flex, "time": 0,  "reconfig_count":self.reconfig_count}
                    results.append(res)
             

        print(f"Number of reconfigurations: {self.reconfig_count}")
        print('full flex')
        return results



        

    def find_resource_request(self, i, s):
        #rough calculation, to be improved
        #print('resource request start')
        
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

            

    
    def find_provision(self, inf_app, single_link_resource_request):
        provision = [[] for _ in range(self.n_apps +1)]
        self.sacrificed_states = [[] for _ in range(self.n_apps+1)]
        self.condi_states = [[] for _ in range(self.n_apps+1)]

        matching_request = [[] for _ in range(self.n_apps+1)]
        matching_request[inf_app] = single_link_resource_request
        #print(single_link_resource_request)

        

        for r in single_link_resource_request:

            for t in [r[1][0], r[1][1]-1]:
                i = self.partition[t][r[0]].app
                if i != -1 and i != inf_app:
                    provision[i].append((r[0],t))

                    #rint(i)

                    if self.current_states[i] not in self.partition[t][r[0]].states:
                        provision[i].remove((r[0],t))
                        
        
                        #self.sacrificed_states[i] += self.partition[t][r[0]].states
                    else:
                        #print('condi:',i)
                        s = self.current_states[i]
                        flow = self.apps[i].states[s].flows
                        a = t//flow[0].period
                        for f in flow:
                            if r[0] in f.txs:
                                a = t//f.period
                        matching_request[i].append((r[0],[a*f.period, (a+1)*f.period]))
                        self.condi_states[i] += self.partition[t][r[0]].states
                
                if i == -1:
                    provision[i].append((r[0],t))

        self.loss = [0]*self.n_apps
        
        for i in range(self.n_apps):
            old_flex = self.calculate_flexibility(i, self.feasible_states[i], self.infeasible_states[i])
            self.sacrificed_states[i] = list(set(self.sacrificed_states[i]))
            new_flex = self.calculate_flexibility(
                i, [s for s in self.feasible_states[i] if s not in self.sacrificed_states[i]]
                , self.infeasible_states[i]+self.sacrificed_states[i])
            self.loss[i] = old_flex-new_flex
        '''
        print(provision)
        print(self.loss)
        print(matching_request)
        '''
        return provision, matching_request
    
    def hungarian_matching(self, provision, loss, matching_request):
        # Prepare a list of valid requests and provisions
        valid_requests = []
        valid_provisions = []
        request_to_app = []
        provision_to_app = []

        # Collect all valid requests
        for req_app in range(len(matching_request)):
            for req in matching_request[req_app]:
                valid_requests.append(req)
                request_to_app.append(req_app)

        # Collect all valid provisions
        for prov_app in range(len(provision)):
            if provision[prov_app]:  # Only consider apps with provisions
                valid_provisions.append(provision[prov_app])
                provision_to_app.append(prov_app)

        # If there are no valid requests or provisions, return no matching
        if not valid_requests or not valid_provisions:
            return None, float('inf')  # Indicate no matching possible

        n_requests = len(valid_requests)
        n_provisions = len(valid_provisions)
        cost_matrix = np.full((n_requests, n_provisions), 9999.0)  # Large value for invalid matches
        

        # Populate the cost matrix
        for req_idx, (req_link, req_time_range) in enumerate(valid_requests):
            for prov_idx, prov_links in enumerate(valid_provisions):
                for prov_link, prov_time in prov_links:
                    if prov_link == req_link and req_time_range[0] <= prov_time <= req_time_range[1]:
                        # Set the cost as the loss of the provider
                        cost_matrix[req_idx][prov_idx] = 0
        
        print('cost matrix:',cost_matrix)

        # Apply the Hungarian algorithm to the valid part of the cost matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Collect the matching result
        matching = []
        total_cost = 0
        for row, col in zip(row_ind, col_ind):
            if cost_matrix[row, col] < 9999:  # Only valid matches
                matching.append((request_to_app[row], provision_to_app[col]))
                total_cost += cost_matrix[row, col]
            else:
                return None, float('inf')  # Indicate that a valid matching was not found

        return matching, total_cost


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

###################################################################################################################################

class Adjustment4:
    def __init__(self, trial, n_apps, T, links, max_n_states, max_n_flows, max_n_flow_hop, verbose):
        self.n_apps = n_apps
        self.T = T
        self.links = links
        self.max_n_states = max_n_states
        self.max_n_flows = max_n_flows
        self.max_n_flow_hop = max_n_flow_hop
        self.verbose = verbose
        self.trial = trial
        random.seed(self.trial)
        self.apps = [App(trial, i, self.links, self.T,
                         max_n_states=self.max_n_states,
                         max_n_flows=self.max_n_flows,
                         max_n_flow_hop=self.max_n_flow_hop)
                     for i in range(self.n_apps)]
        self.reconfig_count = 0
        self.infeasible_times = 0
        self.transition_times = 0
        self.time = 0
        self.n_adjust = 0

    def run(self):
        random.seed(self.trial)
        self.current_states = [random.randint(0, len(self.apps[_].states)-1) for _ in range(self.n_apps)]
        self.heu = Heuristic(0, self.apps, self.links, self.T, self.current_states, verbose=False)
        flex = self.heu.run()
        self.flex = flex
        bt = time.time()
        print('flex:',flex)
        if flex > 0 and flex <= self.n_apps:
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

                        
                        for r in request:
                            for t in [r[1][0], r[1][1]-1]:
                                a = self.partition[t][r[0]].app
                                if a == -1:
                                    self.partition[t][r[0]].app = inf_app
                                    if r in request:
                                        request.remove(r)
                        

                        bt = time.time()
                       
  
                        all_covered = False

                        if all_covered:
                            all_covered = False
 
              
                        if not all_covered:
                            print('not all covered')

                            ac = True
                            request_sorted = sorted(request, key=itemgetter(0))
                            grouped_resource = [list(group) for key, group in groupby(request_sorted, key=itemgetter(0))]

                            total_loss = 0
                        
                            n_app = 0
                            matched_apps = []


                            for l in grouped_resource:
                                print(l)
                                provision, matching_request = self.find_provision(inf_app, single_link_resource_request=l)
                                matching, total_cost = self.hungarian_matching(provision,self.loss, matching_request)
                                total_loss += total_cost
                                #print(matching, total_cost)
                                if matching is None:
                                    ac = False
                                    break
                            
                                for m in matching:
                                    if m[1] not in matched_apps and m[1] != -1:
                                        matched_apps.append(m[1])
                                        n_app += 1

                                    a = m[1]
                                    
                                
                                    for r in provision[a]:
                                        self.partition[r[1]][r[0]].app = inf_app
                                        self.partition[r[1]][r[0]].states = [self.current_states[inf_app]]

                                    self.heu.calculate_flexibility()
                    
       
                            if ac:
                               
                                self.flex = self.heu.calculate_flexibility()
                           

                                self.flex = self.flex-total_loss
                       
                                

                                self.time += time.time()-bt
                    
                                print('Flexibility after matching:', self.flex)
                                res = {"method": "match","n_affected_apps": n_app, "flex": self.flex, "time": time.time()-bt,  "reconfig_count":self.reconfig_count}
                                results.append(res)
                                print('bipartite worked')
                                self.n_adjust = n_app
                                return results
                            
                            if ac == False:
                                self.flex = 0
                                print('bipartite failed')
                                res = {"method": "match","n_affected_apps": 0, "flex": 0, "time": time.time()-bt,  "reconfig_count":self.reconfig_count}
                                results.append(res)
                                return results
                               
                    else:
                        print("still feasible")
                else:
                    print('nothing happened')
                    res = {"method": "reconfig", "n_affected_apps": 0,
                                       "flex": self.flex, "time": 0,  "reconfig_count":self.reconfig_count}
                    results.append(res)
             

        print(f"Number of reconfigurations: {self.reconfig_count}")
        print('full flexibility')

        return results



        

    def find_resource_request(self, i, s):
        #rough calculation, to be improved
        #print('resource request start')
        
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

            

    
    def find_provision(self, inf_app, single_link_resource_request):
        provision = [[] for _ in range(self.n_apps +1)]
        self.sacrificed_states = [[] for _ in range(self.n_apps+1)]
        self.condi_states = [[] for _ in range(self.n_apps+1)]

        matching_request = [[] for _ in range(self.n_apps+1)]
        matching_request[inf_app] = single_link_resource_request
        #print(single_link_resource_request)

        

        for r in single_link_resource_request:

            for t in [r[1][0], r[1][1]-1]:
                i = self.partition[t][r[0]].app
                if i != -1 and i != inf_app:
                    provision[i].append((r[0],t))

                    #rint(i)

                    if self.current_states[i] not in self.partition[t][r[0]].states:
                        provision[i].remove((r[0],t))
                        
        
                        #self.sacrificed_states[i] += self.partition[t][r[0]].states
                    else:
                        #print('condi:',i)
                        s = self.current_states[i]
                        flow = self.apps[i].states[s].flows
                        a = t//flow[0].period
                        for f in flow:
                            if r[0] in f.txs:
                                a = t//f.period
                        matching_request[i].append((r[0],[a*f.period, (a+1)*f.period]))
                        self.condi_states[i] += self.partition[t][r[0]].states
                
                if i == -1:
                    provision[i].append((r[0],t))

        self.loss = [0]*self.n_apps
        
        for i in range(self.n_apps):
            old_flex = self.calculate_flexibility(i, self.feasible_states[i], self.infeasible_states[i])
            self.sacrificed_states[i] = list(set(self.sacrificed_states[i]))
            new_flex = self.calculate_flexibility(
                i, [s for s in self.feasible_states[i] if s not in self.sacrificed_states[i]]
                , self.infeasible_states[i]+self.sacrificed_states[i])
            self.loss[i] = old_flex-new_flex
        '''
        print(provision)
        print(self.loss)
        print(matching_request)
        '''
        return provision, matching_request
    
    def hungarian_matching(self, provision, loss, matching_request):
        # Prepare a list of valid requests and provisions
        valid_requests = []
        valid_provisions = []
        request_to_app = []
        provision_to_app = []

        # Collect all valid requests
        for req_app in range(len(matching_request)):
            for req in matching_request[req_app]:
                valid_requests.append(req)
                request_to_app.append(req_app)

        # Collect all valid provisions
        for prov_app in range(len(provision)):
            if provision[prov_app]:  # Only consider apps with provisions
                valid_provisions.append(provision[prov_app])
                provision_to_app.append(prov_app)

        # If there are no valid requests or provisions, return no matching
        if not valid_requests or not valid_provisions:
            return None, float('inf')  # Indicate no matching possible

        n_requests = len(valid_requests)
        n_provisions = len(valid_provisions)
        cost_matrix = np.full((n_requests, n_provisions), 9999.0)  # Large value for invalid matches
        

        # Populate the cost matrix
        for req_idx, (req_link, req_time_range) in enumerate(valid_requests):
            for prov_idx, prov_links in enumerate(valid_provisions):
                for prov_link, prov_time in prov_links:
                    if prov_link == req_link and req_time_range[0] <= prov_time <= req_time_range[1]:
                        # Set the cost as the loss of the provider
                        cost_matrix[req_idx][prov_idx] = 0
        
        print('cost matrix:',cost_matrix)

        # Apply the Hungarian algorithm to the valid part of the cost matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Collect the matching result
        matching = []
        total_cost = 0
        for row, col in zip(row_ind, col_ind):
            if cost_matrix[row, col] < 9999:  # Only valid matches
                matching.append((request_to_app[row], provision_to_app[col]))
                total_cost += cost_matrix[row, col]
            else:
                return None, float('inf')  # Indicate that a valid matching was not found

        return matching, total_cost


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
    

    


    

# Simulation parameters
n_apps_values = [4,5,6,7,8,9,10,11,12] # Varying number of applications
T = 50
links = range(30)
max_n_states = 10
max_n_flows = 8
max_n_flow_hop = 5


# Function to run simulation for a given Adjustment class
def run_simulation(AdjustmentClass, n_apps,trial):
    inf_l = []
    sr_l = []
    flex_l = []
    par_l = []

    sr = 0
    flex = 0
    n_adjust =0

    i = 0

    trial = 200
    k = 0

    while i < trial:
        adj = AdjustmentClass(k, n_apps=n_apps, T=T, links=links,
                          max_n_states=max_n_states, max_n_flows=max_n_flows,
                          max_n_flow_hop=max_n_flow_hop, verbose=False)
        results = adj.run()
        
        k += 1
        

        # Collect metrics
        if adj.infeasible_times == 1:
            inf_l.append(1)
            i += 1
            if adj.flex != 0:
                sr_l.append(1)
                flex_l.append(adj.flex)
                par_l.append(adj.n_adjust)
        
    if len(sr_l) == 0:
        sr = 0
        flex = 0
        n_adjust = 0
    
    else:
        sr = len(sr_l)/len(inf_l)
        flex = sum(flex_l)/len(flex_l)
        n_adjust = sum(par_l)/len(par_l)

    return {"sr" : sr,
            "avg_flex": flex,
            "adjusted partitions": n_adjust
            }
    

# Data storage for each method
results_heu = []
results_adjust1 = []
results_bipartite = []
results_setbip = []


# Run simulations across different n_apps values with a progress bar
for n_apps in tqdm(n_apps_values, desc="Running simulations for different n_apps"):
    
    results_heu.append(run_simulation(Adjustment1, n_apps, 100))
    results_adjust1.append(run_simulation(Adjustment2, n_apps,100))
    results_setbip.append(run_simulation(Adjustment3, n_apps, 100))
    results_bipartite.append(run_simulation(Adjustment4,n_apps,100))
    


# Plotting the results with markers and custom x-axis ticks
def plot_metric(metric, label):
    plt.figure()
    plt.plot(n_apps_values, [res[metric] for res in results_heu], label="Heuristic ", marker='o')
    plt.plot(n_apps_values, [res[metric] for res in results_adjust1], label="Guaranteed-Only  ", marker='o')
    plt.plot(n_apps_values, [res[metric] for res in results_setbip], label="SetCover+Bipartite", marker='o')
    plt.plot(n_apps_values, [res[metric] for res in results_bipartite], label="Conditional", marker='o')

    plt.xlabel("Number of Applications (n_apps)")
    plt.ylabel(label)
    plt.xticks(n_apps_values)  # Set x-axis ticks to be exactly the n_apps values
    plt.title(f"{label} vs. Number of Applications")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)  # Optional: add grid lines for readability
    plt.show()

plot_metric("sr", "Success Ratio")
plot_metric("avg_flex", "Average Flexibility")
plot_metric("adjusted partitions", 'Number of Adjusted Partitions')

print(results_heu)
print(results_adjust1)
print(results_bipartite)
print(results_setbip)


'''
[{'sr': 0.985, 'avg_flex': 3.5534140609137066, 'adjusted partitions': 4.0}, {'sr': 0.98, 'avg_flex': 3.9637367346938803, 'adjusted partitions': 5.0}, {'sr': 0.975, 'avg_flex': 4.1405409743589745, 'adjusted partitions': 6.0}, {'sr': 0.95, 'avg_flex': 4.020584736842105, 'adjusted partitions': 7.0}, {'sr': 0.905, 'avg_flex': 3.9988345856353593, 'adjusted partitions': 8.0}, {'sr': 0.905, 'avg_flex': 4.009601436464088, 'adjusted partitions': 9.0}, {'sr': 0.85, 'avg_flex': 4.078337882352941, 'adjusted partitions': 10.0}, {'sr': 0.825, 'avg_flex': 4.032144242424243, 'adjusted partitions': 11.0}, {'sr': 0.73, 'avg_flex': 4.212192945205481, 'adjusted partitions': 12.0}]
[{'sr': 0.025, 'avg_flex': 3.0604467371410253, 'adjusted partitions': 1.4}, {'sr': 0.08, 'avg_flex': 2.9533118007653707, 'adjusted partitions': 1.8125}, {'sr': 0.09, 'avg_flex': 3.020263789055481, 'adjusted partitions': 2.2777777777777777}, {'sr': 0.135, 'avg_flex': 2.755597174356177, 'adjusted partitions': 3.074074074074074}, {'sr': 0.205, 'avg_flex': 3.037019318971932, 'adjusted partitions': 3.3658536585365852}, {'sr': 0.21, 'avg_flex': 3.415745834703294, 'adjusted partitions': 3.642857142857143}, {'sr': 0.2, 'avg_flex': 3.4226525028256147, 'adjusted partitions': 4.225}, {'sr': 0.275, 'avg_flex': 3.2632107121345326, 'adjusted partitions': 4.927272727272728}, {'sr': 0.25, 'avg_flex': 3.6517059806403505, 'adjusted partitions': 4.8}]
[{'sr': 0.335, 'avg_flex': 2.846949402985075, 'adjusted partitions': 1.8656716417910448}, {'sr': 0.335, 'avg_flex': 3.479739402985074, 'adjusted partitions': 1.955223880597015}, {'sr': 0.235, 'avg_flex': 4.027311702127657, 'adjusted partitions': 2.3404255319148937}, {'sr': 0.13, 'avg_flex': 4.153463461538461, 'adjusted partitions': 2.6538461538461537}, {'sr': 0.085, 'avg_flex': 4.57630705882353, 'adjusted partitions': 3.176470588235294}, {'sr': 0.05, 'avg_flex': 4.268175, 'adjusted partitions': 2.0}, {'sr': 0.05, 'avg_flex': 3.5584109999999995, 'adjusted partitions': 2.7}, {'sr': 0.06, 'avg_flex': 3.736473333333334, 'adjusted partitions': 3.75}, {'sr': 0.055, 'avg_flex': 4.463111818181818, 'adjusted partitions': 2.272727272727273}]
[{'sr': 0.58, 'avg_flex': 3.3855210662560777, 'adjusted partitions': 3.9224137931034484}, {'sr': 0.595, 'avg_flex': 3.71056847741383, 'adjusted partitions': 4.008403361344538}, {'sr': 0.625, 'avg_flex': 3.9931779856239897, 'adjusted partitions': 4.808}, {'sr': 0.63, 'avg_flex': 3.735435505616008, 'adjusted partitions': 5.611111111111111}, {'sr': 0.615, 'avg_flex': 3.6815246510394255, 'adjusted partitions': 5.67479674796748}, {'sr': 0.635, 'avg_flex': 3.7566408272247114, 'adjusted partitions': 6.362204724409449}, {'sr': 0.67, 'avg_flex': 3.710739553082275, 'adjusted partitions': 7.3059701492537314}, {'sr': 0.73, 'avg_flex': 3.604041432653419, 'adjusted partitions': 7.7534246575342465}, {'sr': 0.75, 'avg_flex': 3.972508393546784, 'adjusted partitions': 8.433333333333334}]
'''