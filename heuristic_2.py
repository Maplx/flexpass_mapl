from app import App
import copy
import numpy as np
import time
from typedefs import *
import random

class Hueristic_2:
    def __init__(self, trial, apps, links, T, current_states, verbose=False):
        self.trial = trial
        self.apps = apps
        self.T = T
        self.links = links
        self.current_states = current_states
        self.verbose = verbose

        self.partition = [[Cell() for e in range(len(links))]
                          for t in range(T)]

        self.schedule = [[Schedule(self.count_txs(i, s)) for s in range(self.apps[i].n_states)]
                         for i in range(len(self.apps))]
        
        self.time_dependency = []
        self.s0_feasibility = True
        self.n = 0
        
        
        

    def run(self):
        for t in range(self.T):
            #initialize flows
            for i in range(len(self.apps)):
                self.time_dependency.append([])
                for s in range(self.apps[i].n_states):
                    self.time_dependency[i].append([])
                    schedule = self.schedule[i][s]
                    schedule.current_used_links = {}
                    flows = self.apps[i].states[s].flows
       
                    for f in flows:
                        self.time_dependency[i][s].append([-1])
                        schedule.packets_to_schedule.append([])
                        if t % f.period == 0:
                            schedule.cur_hops[f.id] = 0

                            for link in f.txs:
                                pkt = Packet(f.id, link, (t//f.period+1)*f.period, f.period)
                                schedule.packets_to_schedule[-1].append(pkt)
                        
                        
       
            
            #current state schedule
        for a in range(len(self.current_states)):
            s = self.current_states[a]
            schedule = self.schedule[a][s]
            fid = 0
            for f in schedule.packets_to_schedule:
                for pkt in f:
                    for e in self.links:
                        if e == pkt.link:
                            for t in range(self.T):
                                if self.partition[t][e].app == -1 and t > self.time_dependency[a][s][fid][-1]:
                                    self.partition[t][e].app = a
                                    schedule.n_packets_scheduled += 1
                                    self.time_dependency[a][s][fid].append(t)
                                    break                       
                fid += 1


            
        flexibility = 0
    
        for i in range(len(self.apps)):
            if self.schedule[i][self.current_states[i]].n_packets_scheduled != self.schedule[i][self.current_states[i]].n_total_packets:
                self.s0_feasibility = False
                break
        if self.s0_feasibility:
            flexibility = self.calculate_flexibility()
            print(flexibility)
        
        else:
            print('cant do it')
            return flexibility
        
        '''
        
        self.l = {}
        for a in range(len(self.apps)):
            for s in self.all_infeasible_states[a]:
             
                self.l[(a,s)] = self.schedule[a][s].n_total_packets
        
        sorted_infeasible_states = sorted(self.l.items(), key=lambda x: x[1])

        for (app_id, state_id), remaining_packets in sorted_infeasible_states:
            schedule = self.schedule[app_id][state_id]
            required_packets = schedule.n_total_packets - schedule.n_packets_scheduled
            feasible = True

            # Attempt to assign required packets
            fid = 0
            for f in schedule.packets_to_schedule:
                for pkt in f:
                    assigned = False
                    for t in range(self.T):
                        if t > self.time_dependency[app_id][state_id][fid][-1] and self.partition[t][pkt.link].app == -1:
                            self.partition[t][pkt.link].app = app_id
                            schedule.n_packets_scheduled += 1
                            self.time_dependency[app_id][state_id][fid].append(t)
                            assigned = True
                            break
                    if not assigned:
                        feasible = False
                        break
                fid += 1
                if not feasible:
                    break
                    '''
        for a in range(len(self.apps)):
            for s in self.all_infeasible_states[a]:
                schedule = self.schedule[a][s]
                fid = 0
                for f in schedule.packets_to_schedule:
                    for pkt in f:
                        assigned = False
                        # Randomly try to assign a slot
                        available_slots = [
                            (t, pkt.link) for t in range(self.T)
                            if self.partition[t][pkt.link].app == -1
                        ]
                        if available_slots:
                            t, link = random.choice(available_slots)  # Choose a random slot
                            self.partition[t][link].app = a
                            schedule.n_packets_scheduled += 1
                            self.time_dependency[a][s][fid].append(t)
                            assigned = True
                        if not assigned:
                            break
                    fid += 1

            
        
        flexibility = self.calculate_flexibility()
        return flexibility


    def count_txs(self, i, s):
        n = 0
        flows = self.apps[i].states[s].flows
        for f in flows:
            n += len(f.txs)*(self.T//f.period)
        return n

    
    def calculate_flexibility(self):
        total_flex = 0
        gamma = 0.9
        self.all_feasible_states = []
        self.all_infeasible_states = []
        for i, app in enumerate(self.apps):
            feasible_states = []
            infeasible_states = []
            for s, sch in enumerate(self.schedule[i]):
                if sch.n_packets_scheduled == sch.n_total_packets:
                    feasible_states.append(s)
                else:
                    infeasible_states.append(s)
            self.all_feasible_states.append(feasible_states)
            self.all_infeasible_states.append(infeasible_states)
            M_prime = copy.deepcopy(app.transitions)
            for s in range(len(M_prime)):
                if s not in feasible_states:
                    M_prime[s, :] = 0
                    M_prime[s, s] = 1
            flex = 0
            denominator = 0
            for k in range(1, app.k_max+1):
                k_step_matrix = np.linalg.matrix_power(M_prime, k)
                k_step_success_prob = sum(k_step_matrix[self.current_states[i], :][s] for s in feasible_states)
                flex += (gamma**k)*k_step_success_prob
                denominator += (gamma**k)
            flex = flex/denominator
            total_flex += flex
        return round(total_flex, 5)

if __name__ == "__main__":
    t = 0
    T = 50
    links = range(30)
    # for t in range(50):
    apps = [App(1, i, links, T=T,
                max_n_states=20,
                max_n_flows=8,
                max_n_flow_hop=5)
            for i in range(10)]
    bt = time.time()
    h = Hueristic_2(t, apps, links, T, current_states=[0]*len(apps))
    h.run()
    print(f"Time {time.time()-bt:.2f}s")
    print(h.calculate_flexibility(),h.s0_feasibility)
    # h.dump()



