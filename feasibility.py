import z3
import numpy as np


class Solver:
    def __init__(self, trial, apps, links, T, verbose=False):
        self.trial = trial
        self.apps = apps
        self.links = links
        self.T = T
        self.verbose = verbose

    def run(self):
        self.solver = z3.Solver()
        self.solver.add(self.constraint_schedule())
        res = self.solver.check()
        if self.verbose:
            print(f"Trial {self.trial}: {res}")

        return self.dump(res)

    def constraint_schedule(self):
        constraints = []

        all_tx_vars = []
        for i in range(len(self.apps)):
            flows = self.apps[i].states[0].flows
            for f in flows:
                for k in range(self.T//f.period):
                    tx_vars = []
                    for hop, e in enumerate(f.txs):
                        tx_var = z3.Int(f"tx_{i}_{f.id}_{k}_{hop}_{e}")
                        tx_vars += [tx_var]
                        all_tx_vars += [tx_var]

                        # precedence
                        if hop > 0:
                            constraints += [tx_vars[hop] > tx_vars[hop-1]]
                        # timing
                        constraints += [k*f.period <= tx_var, tx_var < (k+1)*f.period]

        for e in self.links:
            conflict_txs = [tx_var for tx_var in all_tx_vars if tx_var.decl().name().split("_")[-1] == f"{e}"]
            if conflict_txs:
                constraints += [z3.Distinct(conflict_txs)]

        constraints += [z3.simplify(z3.And(constraints))]
        return constraints

    def dump(self, res):
        if res != z3.sat:
            return None

        result = self.solver.model()

        partition = [[-1 for e in self.links]
                     for t in range(self.T)]

        for var in result:
            i = int(str(var).split('_')[1])
            e = int(str(var).split('_')[-1])
            t = result[var].as_long()
            partition[t][e] = i
        return partition
