class Gain:
    def __init__(self, app):
        self.app = app
        self.pkts_per_state = []
        self.n_tx_per_state = []
        self.value = 0


class Progress:
    def __init__(self):
        self.current = 0
        self.total = 0


class Cell:
    def __init__(self):
        self.app = -1
        self.states = []


class Schedule:
    def __init__(self, total):
        self.current_used_links = {}
        self.packets_to_schedule: list[Packet] = []
        self.n_packets_scheduled = 0
        self.n_total_packets = total
        self.cur_hops = {}


class Packet:
    def __init__(self, flow_id, link, deadline, period):
        self.flow_id = flow_id
        self.link = link
        self.deadline = deadline
        self.period = period


class State:
    def __init__(self, id):
        self.id = id
        self.flows: list[Flow] = []


class Flow:
    def __init__(self, id, txs, period):
        self.id = id
        self.txs = txs
        self.period = period
