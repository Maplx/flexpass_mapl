import z3
import numpy as np
import json


class Optimizer:
    def __init__(self, trial, apps, links, T, verbose=False):
        self.trial = trial
        self.apps = apps
        self.links = links
        self.T = T
        self.verbose = verbose
        self.dump_json = True

    def run(self):
        self.opt = z3.Optimize()
        self.opt.set("timeout", 60*60*1000)
        self.create_vars()
        self.opt.add(self.constraint_conflict())
        self.opt.add(self.constraint_states_feasibility())
        self.objective_flexibility()
        res = self.opt.check()
        total_flexibility = 0
        if self.verbose:
            print(f"Trial {self.trial}: {res}")

        total_flexibility = self.dump(res)

        return total_flexibility

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
        if res != z3.sat:
            if self.verbose:
                for i in range(len(self.apps)):
                    print(f'[A_{i}] \n\tstates: {self.apps[i].states}\n\ttransitions:{self.apps[i].transitions}')
            return 0

        result = self.opt.model()
        D = {
            "T": self.T,
            "links": [e for e in self.links],
            "apps": [{
                "id": app.id,
                "n_states": app.n_states,
                "states": app.states,
                "transitions": app.transitions,
                "k_max": app.k_max,
                "satisfied_states": [],
                "flexibility": 0
            } for app in self.apps],
            "flexibilities": [],
            "partitions": [],
            "schedule": {},
        }
        total_flex = 0
        for i in range(len(self.apps)):
            partition = []
            for t in range(self.T):
                e_set = []
                for e in range(len(self.links)):
                    if result[self.X[t][e][i]] == True:
                        e_set += [self.links[e]]
                if e_set:
                    partition += [(t, e_set)]
            D["partitions"] += [partition]
            D["apps"][i]["flexibility"] = round(float(result[self.Phi[i]].as_decimal(8).replace("?", "")), 4)
            total_flex += D["apps"][i]["flexibility"]
            for s in range(self.apps[i].n_states):
                if result[self.F[i]].as_long() >> s & 1 == 1:
                    D["apps"][i]["satisfied_states"] += [s]

            if self.verbose:
                print(f'[A_{i}]')
                print(f'\tstates: {self.apps[i].states}')
                print('\ttransition matrix:')
                print('\n'.join('\t\t'+'[' +
                      ', '.join(map(str, row)) + ']' for row in self.apps[i].transitions))
                print(f'\tsatisfied states: {D["apps"][i]["satisfied_states"]}')
                print(f'\tk_max: {self.apps[i].k_max}, gamma: {self.gamma}, flexibility: {D["apps"][i]["flexibility"]}')
                # print(f'\tpartition: {partitions[i]}\n')
        if self.verbose:
            print(f"total flexibilities: {round(total_flex, 4)}")
        if self.dump_json:
            for v in result:
                if str(v)[:2] == "tx":
                    D["schedule"][str(v)] = result[v].as_long()
            # with open(f'./flex-vis/src/assets/smt/res-{self.trial}.json', 'w') as outfile:
            #     json.dump(D, outfile, cls=NumpyEncoder)
        return round(total_flex, 4)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
