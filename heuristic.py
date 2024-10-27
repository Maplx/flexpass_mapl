from app import App
import copy
import numpy as np
import time
from typedefs import *


class Heuristic:
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

    def run(self):
        for t in range(self.T):
            for i in range(len(self.apps)):
                for s in range(self.apps[i].n_states):
                    schedule = self.schedule[i][s]
                    schedule.current_used_links = {}
                    flows = self.apps[i].states[s].flows
                    for f in flows:
                        if t % f.period == 0:
                            schedule.cur_hops[f.id] = 0
                            pkt = Packet(f.id, f.txs[schedule.cur_hops[f.id]], (t//f.period+1)*f.period, f.period)
                            schedule.packets_to_schedule.append(pkt)

                    schedule.packets_to_schedule.sort(key=lambda x: (x.deadline, x.flow_id))
            for e in self.links:
                if self.partition[t][e].app == -1:
                    gains: list[Gain] = []
                    for i in range(len(self.apps)):
                        gains.append(self.potenial_gain(i, e, t))
                    gains.sort(key=lambda g: (g.value, g.app), reverse=True)

                    if gains[0].value > 0:
                        app = gains[0].app

                        self.partition[t][e].app = app

                        for s in range(self.apps[app].n_states):
                            schedule: Schedule = self.schedule[app][s]
                            schedule.n_packets_scheduled += len(gains[0].pkts_per_state[s])
                            if len(gains[0].pkts_per_state[s]) > 0:
                                self.partition[t][e].states.append(s)
                            schedule.packets_to_schedule = list(
                                set(schedule.packets_to_schedule)-set(gains[0].pkts_per_state[s]))
                            schedule.packets_to_schedule.sort(key=lambda x: (x.deadline, x.flow_id))
                            for pkt in gains[0].pkts_per_state[s]:
                                schedule.current_used_links[pkt.link] = True
                                schedule.cur_hops[pkt.flow_id] += 1
                                f = self.apps[app].states[s].flows[pkt.flow_id]
                                if schedule.cur_hops[pkt.flow_id] < len(f.txs):
                                    pkt = Packet(f.id, f.txs[schedule.cur_hops[f.id]],
                                                 (t//f.period+1)*f.period, f.period)
                                    schedule.packets_to_schedule.append(pkt)

        flexibility = 0
        s0_feasibility = True
        for i in range(len(self.apps)):
            if self.schedule[i][self.current_states[i]].n_packets_scheduled != self.schedule[i][self.current_states[i]].n_total_packets:
                s0_feasibility = False
                break
        if s0_feasibility:
            flexibility = self.calculate_flexibility()

        return flexibility

    def count_txs(self, i, s):
        n = 0
        flows = self.apps[i].states[s].flows
        for f in flows:
            n += len(f.txs)*(self.T//f.period)
        return n

    def potenial_gain(self, i, e, t):
        self.partition[t][e].app = i
        gain = Gain(i)
        gamma = 0.9
        for s in range(self.apps[i].n_states):
            if (self.schedule[i][s].packets_to_schedule) == 0:
                continue
            gain.pkts_per_state += [self.check_new_scheduled_packets(t, i, s)]
            feasibility_gain = len(gain.pkts_per_state[s]) / self.schedule[i][s].n_total_packets
            weight = 0
            for k in range(1, self.apps[i].k_max+1):
                weight += gamma**k*self.apps[i].M_k[k][s]
            # higher weight for initial state
            if s == self.current_states[i]:
                weight = weight*20
            gain.value += feasibility_gain*weight

        self.partition[t][e].app = -1
        return gain

    def check_new_scheduled_packets(self, t, i, s):
        schedule = self.schedule[i][s]
        used_links = copy.deepcopy(schedule.current_used_links)
        new_scheduled_packets = []
        for pkt in schedule.packets_to_schedule:
            if self.partition[t][pkt.link].app == i and pkt.link not in used_links:
                used_links[pkt.link] = True
                new_scheduled_packets += [pkt]
        return new_scheduled_packets

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
    T = 40
    links = range(50)
    # for t in range(50):
    apps = [App(t, i, links, T=T,
                max_n_states=20,
                max_n_flows=6,
                max_n_flow_hop=4)
            for i in range(10)]
    bt = time.time()
    h = Heuristic(t, apps, links, T, current_states=[0]*len(apps))
    h.run()
    print(f"Time {time.time()-bt:.2f}s")
    print(h.calculate_flexibility())
    # h.dump()
