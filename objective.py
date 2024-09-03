import z3
import numpy as np
from app import App
import time
from concurrent.futures import ProcessPoolExecutor

OBJ_FLEX = 0
OBJ_N_STATES = 1
OBJ_MAX_PROB = 2


class Optimizer:
    def __init__(self, trial, apps, links, T, objective=OBJ_FLEX):
        self.trial = trial
        self.apps = apps
        self.links = links
        self.T = T
        self.objective = objective
        self.dump_json = True

    def run(self):
        self.opt = z3.Optimize()
        self.create_vars()
        self.opt.add(self.constraint_conflict())
        self.opt.add(self.constraint_states_feasibility())

        if self.objective == OBJ_FLEX:
            self.objective_flexibility()
        elif self.objective == OBJ_N_STATES:
            self.objective_n_states()
        elif self.objective == OBJ_MAX_PROB:
            self.objective_max_stationary_prob()
        res = self.opt.check()

        return self.dump(res)

    def create_vars(self):
        # allocation of each link-time to app
        self.X = [[[z3.Bool(f'x_{t}_{e}_{i}')
                    for i in range(len(self.apps))]
                   for e in self.links]
                  for t in range(self.T)]

        # state feasibility
        self.F = [z3.BitVec(f"F_{i}", self.apps[i].n_states)
                  for i in range(len(self.apps))]

        # k-step success probability
        self.KP = [[z3.Array(f'KP_{i}_{k}', z3.BitVecSort(self.apps[i].n_states), z3.RealSort())
                    for k in range(self.apps[i].k_max+1)]
                   for i in range(len(self.apps))]

        # flexibility score
        self.Phi = [z3.Real(f"Phi_{i}") for i in range(len(self.apps))]

    def constraint_conflict(self):
        constraints = []
        for t in range(self.T):
            for e in range(len(self.links)):
                constraints += [z3.AtMost(*self.X[t][e], 1)]
        return constraints

    def constraint_states_feasibility(self):
        constraints = []
        for i in range(len(self.apps)):
            for s in range(self.apps[i].n_states):
                self.opt.add(self.constraint_state_feasibility(i, s))
                if s == 0:
                    self.opt.add((self.F[i] >> s) & 1 == 1)
                else:
                    self.opt.add_soft((self.F[i] >> s) & 1 == 1)
        return constraints

    def constraint_state_feasibility(self, i, s):
        constraints = []
        sch_constraints = []
        flows = self.apps[i].states[s].flows
        all_tx_vars = []
        for f in flows:
            for k in range(self.T//f.period):
                tx_vars = []
                for hop, e in enumerate(f.txs):
                    tx_var = z3.Int(f"tx_{i}_{s}_{f.id}_{k}_{hop}_{e}")
                    tx_vars += [tx_var]
                    all_tx_vars += [tx_var]

                    # precedence
                    if hop > 0:
                        sch_constraints += [tx_vars[hop] > tx_vars[hop-1]]
                    # timing
                    sch_constraints += [k*f.period <= tx_var, tx_var < (k+1)*f.period]
                    # match partition allocation
                    for t in range(k*f.period,  (k+1)*f.period):
                        sch_constraints += [z3.Implies(tx_var == t, self.X[t][e][i])]

        for e in self.links:
            conflict_txs = [tx_var for tx_var in all_tx_vars if tx_var.decl().name().split("_")[-1] == f"{e}"]
            if conflict_txs:
                sch_constraints += [z3.Distinct(conflict_txs)]

        constraints += [z3.Implies((self.F[i] >> s) & 1 == 1, z3.simplify(z3.And(sch_constraints)))]
        return constraints

    def objective_flexibility(self):
        self.gamma = 0.9
        for i in range(len(self.apps)):
            for bit_vec in range(2**self.apps[i].n_states):
                for k in range(1, self.apps[i].k_max+1):
                    self.calculate_k_step_success_prob(i, bit_vec, k)

            self.opt.add(self.Phi[i] == sum(z3.Select(self.KP[i][k], self.F[i])*(self.gamma**k)
                         for k in range(1, self.apps[i].k_max+1))/sum(self.gamma**k for k in range(1, self.apps[i].k_max+1)))
        self.opt.maximize(z3.Sum(self.Phi))

    def objective_n_states(self):
        for i in range(len(self.apps)):
            n = z3.Real(f"n_{i}")
            self.opt.add(n == z3.Sum([self.F[i] >> s & 1 == 1
                         for s in range(self.apps[i].n_states)]))
            self.opt.add(self.Phi[i] == n/self.apps[i].n_states)
        self.opt.maximize(z3.Sum(self.Phi))

    def objective_max_stationary_prob(self):
        for i in range(len(self.apps)):
            self.opt.add(self.Phi[i] == z3.Sum([z3.If(self.F[i] >> s & 1 == 1, self.apps[i].steady[s], 0)
                         for s in range(self.apps[i].n_states)]))
        self.opt.maximize(z3.Sum(self.Phi))

    def calculate_k_step_success_prob(self, i, bit_vec, k):
        feasible_states = []
        for s in range(self.apps[i].n_states):
            if bit_vec >> s & 1 == 1:
                feasible_states += [s]

        M = np.copy(self.apps[i].transitions)
        for s in range(len(M)):
            if s not in feasible_states:
                M[s, :] = 0
                M[s, s] = 1

        k_step_matrix = np.linalg.matrix_power(M, k)

        success_prob = np.sum(k_step_matrix[0, :][s] for s in feasible_states)
        self.KP[i][k] = z3.Store(self.KP[i][k], bit_vec, success_prob)

    def dump(self, res):
        feasible_states = {}
        for i in range(len(self.apps)):
            feasible_states[i] = []

        if res == z3.sat:
            result = self.opt.model()
            for i in range(len(self.apps)):
                # print(result[self.Phi[i]])
                # print(round(float(result[self.Phi[i]].as_decimal(8).replace("?", "")), 4))
                for s in range(self.apps[i].n_states):
                    if result[self.F[i]].as_long() >> s & 1 == 1:
                        feasible_states[i] += [s]

        return feasible_states


class Trial:
    def __init__(self, n_apps, T, links, max_n_states, max_n_flows, max_n_flow_hop, objective):
        self.n_apps = n_apps
        self.T = T
        self.links = links
        self.max_n_states = max_n_states
        self.max_n_flows = max_n_flows
        self.max_n_flow_hop = max_n_flow_hop
        self.objective = objective

    def __call__(self, t):
        apps = [App(t, i, self.links, self.T,
                    max_n_states=self.max_n_states,
                    max_n_flows=self.max_n_flows,
                    max_n_flow_hop=self.max_n_flow_hop)
                for i in range(self.n_apps)]
        # print(apps)
        opt = Optimizer(t, apps, self.links, self.T, self.objective)
        feasible_states = opt.run()
        # print(feasible_states)
        # monte carlo run
        n_sim = 1000
        ret = []
        for steps in range(1, 40):
            reconfig_cnt = 0
            for _ in range(n_sim):
                current_state = [0]*self.n_apps
                next_state = [0]*self.n_apps
                for _ in range(steps):
                    need_reconfig = False
                    for i in range(self.n_apps):
                        next_state[i] = np.random.choice(
                            range(apps[i].n_states), p=apps[i].transitions[current_state[i]])
                        if next_state[i] not in feasible_states[i]:
                            need_reconfig = True
                            break
                        current_state[i] = next_state[i]
                    if need_reconfig:
                        reconfig_cnt += 1
                        break
            ret += [reconfig_cnt/n_sim*100]

        return (t, ret)


if __name__ == "__main__":
    n_trials = 20
    bt = time.time()

    for obj in [OBJ_FLEX, OBJ_N_STATES, OBJ_MAX_PROB]:
        tt = time.time()
        trial = Trial(n_apps=4, T=10, links=range(40),
                      max_n_states=10, max_n_flows=15, max_n_flow_hop=5, objective=obj)

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(trial, range(n_trials)))

        probs = []
        for steps in range(len(results[0][1])):
            prob = 0
            for r in results:
                prob += r[1][steps]/n_trials
            probs += [round(prob, 4)]
        print(probs)

    print('time elapsed: %.2fs' % (time.time()-bt))
