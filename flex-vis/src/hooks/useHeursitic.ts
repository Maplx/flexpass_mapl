import { ref } from 'vue'
import { SeededRandom } from './rand'
const Rand = new SeededRandom(1234)

interface App {
  a_id: number
  states: State[]
  transitions: number[][]
}

interface State {
  s_id: number
  flows: Flow[]
}

interface Flow {
  f_id: number
  txs: number[]
  period: number
}

export function useHeuristic() {
  const E = 10
  const T = 10

  const apps = generateApps(4, E, T)
  const partition = ref<any>([])

  for (let e = 0; e < E; e++) {
    const row: any = { link: e, slots: [] }
    for (let t = 0; t < T; t++) {
      row.slots[t] = { app: -1, states: [], is_checkpoint: false }
    }
    partition.value.push(row)
  }

  const progress = ref<any>([])
  for (let i = 0; i < apps.length; i++) {
    progress.value[i] = { app: i, states: [] }
    for (let s = 0; s < apps[i].states.length; s++) {
      let total = 0
      for (const f of apps[i].states[s].flows) {
        total += (T / f.period) * f.txs.length
      }
      progress.value[i].states[s] = { cur: 0, total: total }
    }
  }

  const all_feasible_states = ref<string[]>([])
  const new_feasible_states = ref<string[]>([])
  const flexibility = ref(0)
  const cur_slot = ref(0)
  const cur_link = ref(0)
  const cur_gains = ref<any>([])
  const cur_pass_id = ref(0)
  const check_points = ref<any>({ 0: [] })
  const black_list = ref<any>({})
  for (let i = 0; i < apps.length; i++) {
    black_list.value[i] = []
  }

  const pass = () => {
    while (cur_slot.value < T) {
      step()
    }
    cur_slot.value = 0
    console.log(check_points.value[cur_pass_id.value].length)
  }

  const step = () => {
    new_feasible_states.value = []
    cur_gains.value = []
    if (partition.value[cur_link.value].slots[cur_slot.value].app == -1) {
      for (let i = 0; i < apps.length; i++) {
        cur_gains.value.push(potential_gain(i))
      }
      cur_gains.value.sort((a: any, b: any) => -a.total + b.total)
      const gain = cur_gains.value[0]

      if (gain.total > 0) {
        const ss = []
        partition.value[cur_link.value].slots[cur_slot.value].app = gain.app
        for (let s = 0; s < gain.n_tx_by_states.length; s++) {
          progress.value[gain.app].states[s].cur += gain.n_tx_by_states[s]
          if (gain.n_tx_by_states[s] > 0) {
            partition.value[cur_link.value].slots[cur_slot.value].states.push(s)
          }

          if (
            !all_feasible_states.value.includes(`${gain.app}-${s}`) &&
            progress.value[gain.app].states[s].cur == progress.value[gain.app].states[s].total
          ) {
            new_feasible_states.value.push(`${gain.app}-${s}`)
            all_feasible_states.value.push(`${gain.app}-${s}`)
            ss.push(s)
          }
        }
        if (new_feasible_states.value.length > 0) {
          partition.value[cur_link.value].slots[cur_slot.value].is_checkpoint = true
          check_points.value[cur_pass_id.value].push({
            t: cur_slot.value,
            e: cur_link.value,
            app: gain.app,
            states: ss,
            partition: JSON.parse(JSON.stringify(partition.value)),
            progress: JSON.parse(JSON.stringify(progress.value))
          })
        }
      }

      calculate_flexibility()
    }
    cur_link.value++
    if (cur_link.value == E) {
      cur_link.value = 0
      cur_slot.value++
    }
  }

  const potential_gain = (i: number): any => {
    const partition_tmp = JSON.parse(JSON.stringify(partition.value))
    partition_tmp[cur_link.value].slots[cur_slot.value].app = i
    const gain = { app: i, total: 0, n_tx_by_states: <number[]>[] }
    for (let s = 0; s < apps[i].states.length; s++) {
      if (black_list.value[i].includes(s)) {
        gain.n_tx_by_states.push(0)
      } else {
        const n_scheduled_txs = check_scheduled_txs(i, s, partition_tmp)
        const g = n_scheduled_txs - progress.value[i].states[s].cur
        gain.n_tx_by_states.push(g)
        gain.total += g / progress.value[i].states[s].total
      }
    }
    gain.total = parseFloat(gain.total.toFixed(5))
    return gain
  }

  const check_scheduled_txs = (i: number, s: number, partition: any): number => {
    let n_scheduled_txs = 0
    const cur_hop: any = {}
    const flows = apps[i].states[s].flows
    for (let t = 0; t < T; t++) {
      const to_be_scheduled: any = []
      for (const f of flows) {
        if (t % f.period == 0) {
          cur_hop[f.f_id] = 0
        }
        if (cur_hop[f.f_id] < f.txs.length) {
          to_be_scheduled.push({
            f_id: f.f_id,
            e: f.txs[cur_hop[f.f_id]],
            deadline: (Math.floor(t / f.period) + 1) * f.period
          })
        }
      }
      to_be_scheduled.sort((a: any, b: any) => a.deadline - b.deadline)
      const e_accessed: any = {}

      for (const tx of to_be_scheduled) {
        if (partition[tx.e].slots[t].app == i && e_accessed[tx.e] == undefined) {
          e_accessed[tx.e] = true
          cur_hop[tx.f_id]++
          n_scheduled_txs++
        }
      }
    }

    return n_scheduled_txs
  }

  const revoke = () => {
    const pointIndex = Math.floor(1)
    const point = check_points.value[cur_pass_id.value][pointIndex]
    console.log(check_points.value[cur_pass_id.value].length, pointIndex)
    if (point == undefined) return

    check_points.value[cur_pass_id.value + 1] = check_points.value[cur_pass_id.value].slice(
      0,
      pointIndex
    )
    cur_pass_id.value++

    partition.value = point.partition
    progress.value = point.progress

    for (let i = 0; i < apps.length; i++) {
      black_list.value[i] = []
    }
    for (const s of point.states) {
      black_list.value[point.app].push(s)
      progress.value[point.app].states[s].cur = 0
    }

    all_feasible_states.value = []
    for (let i = 0; i < apps.length; i++) {
      for (let s = 0; s < apps[i].states.length; s++) {
        if (progress.value[i].states[s].cur == progress.value[i].states[s].total) {
          all_feasible_states.value.push(`${i}-${s}`)
        }
      }
    }

    for (let e = 0; e < E; e++) {
      for (let t = 0; t < T; t++) {
        if (partition.value[e].slots[t].app == point.app) {
          partition.value[e].slots[t].states = partition.value[e].slots[t].states.filter(function (
            item: number
          ) {
            return !point.states.includes(item)
          })
          if (partition.value[e].slots[t].states.length == 0) {
            partition.value[e].slots[t].app = -1
          }
        }
      }
    }
  }

  const calculate_flexibility = () => {
    flexibility.value = 0
    const gamma = 0.9
    const k_max = 6
    for (let i = 0; i < apps.length; i++) {
      if (progress.value[i].states[0].cur == progress.value[i].states[0].total) {
        let prob = 0
        let denominator = 0
        for (let k = 1; k <= k_max; k++) {
          prob += Math.pow(gamma, k) * kStepSuccessProb(i, k)
          denominator += Math.pow(gamma, k)
        }
        flexibility.value += parseFloat((prob / denominator).toFixed(4))
      }
    }
  }

  const kStepSuccessProb = (i: number, k: number): number => {
    const M = JSON.parse(JSON.stringify(apps[i].transitions))

    for (let s = 0; s < apps[i].states.length; s++) {
      if (progress.value[i].states[s].cur != progress.value[i].states[s].total) {
        for (let ss = 0; ss < apps[i].states.length; ss++) {
          if (s == ss) {
            M[s][ss] = 1
          } else {
            M[s][ss] = 0
          }
        }
      }
    }

    const M_k = matrixPower(M, k)
    const prob = M_k[0].reduce(
      (sum: number, v: number, s: number) =>
        progress.value[i].states[s].cur == progress.value[i].states[s].total ? sum + v : sum,
      0
    )
    return prob
  }
  const matrixPower = (matrix: number[][], k: number): number[][] => {
    let result: number[][] = matrix.map((row, i) => row.map((_, j) => (i === j ? 1 : 0))) // Identity matrix
    let tempMatrix = matrix

    while (k > 0) {
      if (k % 2 === 1) {
        // Multiply result by tempMatrix
        result = result.map((row, i) =>
          row.map((_, j) => row.reduce((sum, _, n) => sum + result[i][n] * tempMatrix[n][j], 0))
        )
      }
      // Square the matrix
      tempMatrix = tempMatrix.map((row, i) =>
        row.map((_, j) => row.reduce((sum, _, n) => sum + tempMatrix[i][n] * tempMatrix[n][j], 0))
      )

      k = Math.floor(k / 2)
    }

    return result
  }

  return {
    E,
    T,
    apps,
    partition,
    flexibility,
    all_feasible_states,
    new_feasible_states,
    check_points,
    progress,
    step,
    pass,
    revoke,
    cur_slot,
    cur_link,
    cur_gains,
    black_list
  }
}

function generateApps(n: number, E: number, T: number): App[] {
  const apps: App[] = []
  const max_n_states = 4
  const max_n_flows = 4
  const max_n_hops = 4

  for (let i = 0; i < n; i++) {
    const n_states = 2 + Math.round(Rand.next() * (max_n_states - 2))
    const states: State[] = []
    for (let s = 0; s < n_states; s++) {
      const n_flows = 1 + Math.round(Rand.next() * (max_n_flows - 1))
      const flows: Flow[] = []
      for (let f = 0; f < n_flows; f++) {
        const n_hops = 1 + Math.round(Rand.next() * (max_n_hops - 1))
        const txs: number[] = []
        for (let h = 0; h < n_hops; h++) {
          txs.push(Math.round(Rand.next() * (E - 1)))
        }
        flows.push({ f_id: f, txs: txs, period: [T / 2, T][Math.round(Rand.next())] })
      }
      states.push({ s_id: s, flows: flows })
    }

    const transitions: any = []
    for (let i = 0; i < n_states; i++) {
      transitions.push([])
      let remain_prob = 1
      for (let j = 0; j < n_states - 1; j++) {
        const prob = parseFloat((Rand.next() * remain_prob).toFixed(2))
        transitions[i].push(prob)
        remain_prob = parseFloat((remain_prob - prob).toFixed(2))
      }
      transitions[i].push(remain_prob)
      const rowSum = transitions[i].reduce((acc: number, num: number) => acc + num, 0)
      const roundingError = parseFloat((1 - rowSum).toFixed(2))
      transitions[i][n_states - 1] = transitions[i][n_states - 1] = parseFloat(
        roundingError + transitions[i][n_states - 1].toFixed(2)
      )
    }

    apps.push({ a_id: i, states: states, transitions: transitions })
  }
  return apps
}
