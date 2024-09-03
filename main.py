from app import App
from optimizer import Optimizer
from heuristic import Heuristic
from concurrent.futures import ProcessPoolExecutor
import time


class Trial:
    def __init__(self, n_apps, T, links, max_n_states, max_n_flows, max_n_flow_hop, verbose=False):
        self.n_apps = n_apps
        self.T = T
        self.links = links
        self.max_n_states = max_n_states
        self.max_n_flows = max_n_flows
        self.max_n_flow_hop = max_n_flow_hop
        self.verbose = verbose

    def __call__(self, t):
        apps = [App(t, i, self.links, self.T,
                    max_n_states=self.max_n_states,
                    max_n_flows=self.max_n_flows,
                    max_n_flow_hop=self.max_n_flow_hop)
                for i in range(self.n_apps)]
        bt = time.time()
        # opt = Optimizer(t, apps, self.links, self.T, self.verbose)
        # return (t, opt.run(), time.time()-bt)

        heu = Heuristic(t, apps, self.links, self.T, [0]*len(apps),  verbose=self.verbose)
        return (t, heu.run(), time.time()-bt)


if __name__ == "__main__":
    n_trials = 1
    bt = time.time()
    success_ratio = []
    flexibility = []
    x_axis = [x for x in range(20, 21)]

    for i in x_axis:
        tt = time.time()
        trial = Trial(n_apps=10, T=50, links=range(30),
                      max_n_states=i, max_n_flows=8, max_n_flow_hop=5,
                      verbose=False)

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(trial, range(n_trials)))
        sr = sum(1 for res in results if res[1] > 0)/n_trials*100
        avg_time = sum(res[2] for res in results)/n_trials
        if sr > 0:
            # flex = round(sum(res[1] for res in results)/sum(1 for res in results if res[1] > 0), 4)
            flex = round(sum(res[1] for res in results)/n_trials, 4)
        else:
            flex = 0
        print(f"[+] n_states: {i}, sr: {sr}%, flex: {flex}, average time elapsed: {avg_time:.4f}s")
        success_ratio += [sr]
        flexibility += [flex]
    print('x axis:', x_axis)
    print('success ratio:', success_ratio)
    print('flexibility:', flexibility)
