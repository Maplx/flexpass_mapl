import * as echarts from 'echarts'
import { ref, watch } from 'vue'
import { SeededRandom } from './rand'

const Rand = new SeededRandom(+new Date())
// const Rand = new SeededRandom(17)

export function useMarkovChain() {
  const n_state = 5
  const nodes: any = []
  const edges: any = []
  const transitions: any = []
  const k = ref(1)
  const gamma = ref(0.9)
  const kStepTransition = ref<number[][]>([])
  const feasible_states = ref([0])
  const flexibility = ref(0)

  for (let i = 0; i < n_state; i++) {
    nodes.push({ name: `${i}`, itemStyle: { color: 'red' } })
  }
  nodes[0].itemStyle.color = 'lime'

  for (let i = 0; i < n_state; i++) {
    transitions.push([])
    let remain_prob = 1
    for (let j = 0; j < n_state - 1; j++) {
      const prob = parseFloat((Rand.next() * remain_prob).toFixed(2))
      transitions[i].push(prob)
      remain_prob = parseFloat((remain_prob - prob).toFixed(2))
    }
    transitions[i].push(remain_prob)
    const rowSum = transitions[i].reduce((acc: number, num: number) => acc + num, 0)
    const roundingError = parseFloat((1 - rowSum).toFixed(2))
    transitions[i][n_state - 1] = transitions[i][n_state - 1] = parseFloat(
      roundingError + transitions[i][n_state - 1].toFixed(2)
    )
  }

  for (let i = 0; i < n_state; i++) {
    for (let j = 0; j < n_state; j++) {
      if (i != j && transitions[i][j] > 0) {
        let curveness = 0.1
        if (i < j && transitions[j][i] > 0) {
          curveness = -0.1
        } else if (transitions[j][i] == 0) {
          curveness = 0
        }
        edges.push({
          value: `${transitions[i][j]}`,
          source: i,
          target: j,
          lineStyle: { width: 1.2, color: 'black', opacity: 1, curveness: curveness }
        })
      }
    }
  }
  const probs: any = {}
  const kStepSuccessProb = (feasible_states: number[], s0: number, k: number): number => {
    const M = JSON.parse(JSON.stringify(transitions))
    kStepTransition.value = M
    // set infeasible as absorbing state
    for (let s = 0; s < n_state; s++) {
      if (!feasible_states.includes(s)) {
        for (let ss = 0; ss < n_state; ss++) {
          if (s == ss) {
            M[s][ss] = 1
          } else {
            M[s][ss] = 0
          }
        }
      }
    }

    kStepTransition.value = matrixPower(M, k)
    const prob = kStepTransition.value[s0].reduce(
      (sum, v, i) => (feasible_states.includes(i) ? sum + v : sum),
      0
    )
    if (probs[JSON.stringify(feasible_states)] == undefined) {
      probs[JSON.stringify(feasible_states)] = [0, prob]
    } else {
      probs[JSON.stringify(feasible_states)].push(prob)
    }
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
  const drawGraph = (chartDom: HTMLElement) => {
    const option = {
      series: [
        {
          type: 'graph',
          layout: 'circular',
          animation: false,
          symbolSize: 24,
          edgeSymbol: ['none', 'arrow'],
          edgeSymbolSize: 6,
          label: { show: true },
          edgeLabel: { show: true, formatter: '{c}' },
          nodes: nodes,
          edges: edges
        }
      ]
    }
    const chart = echarts.init(chartDom)
    chart.setOption(option)

    // toggle feasible state
    chart.on('click', function (params: any) {
      if (params.dataType == 'node' && params.name != '0') {
        if (nodes[params.name].itemStyle.color == 'lime') {
          nodes[params.name].itemStyle.color = 'red'
        } else {
          nodes[params.name].itemStyle.color = 'lime'
        }

        feasible_states.value = []
        for (const s of nodes) {
          if (s.itemStyle.color == 'lime') {
            feasible_states.value.push(parseInt(s.name))
          }
        }

        chart.setOption(option)
      }
    })
  }
  watch(k, () => {
    flexibility.value = kStepSuccessProb(feasible_states.value, 0, k.value)
  })

  const drawLine = (chartDom: HTMLElement) => {
    const K: number[] = []
    for (let i = 1; i <= 100; i++) {
      K.push(i)
    }
    const option: any = {
      toolbox: {
        feature: {
          saveAsImage: {
            pixelRatio: 8
          }
        }
      },

      legend: {
        width: '10%',
        top: 80,
        right: 60
      },
      grid: {
        top: '60px',
        left: '70px',
        right: '60px',
        bottom: '20px'
        // containLabel: true
      },
      yAxis: [
        {
          name: 'K-step Success Prob',
          type: 'value',
          max: 1,
          nameTextStyle: { fontSize: 14 }
        }
      ],
      xAxis: {
        name: 'k-step',
        type: 'category',
        data: K
      },
      series: []
    }

    const chart = echarts.init(chartDom)

    for (const f_states of generatePermutations([0, 1, 2, 3, 4])) {
      const data = []
      for (const kk of K) {
        data.push(parseFloat(kStepSuccessProb(f_states, 0, kk).toFixed(8)))
      }
      option.series.push({
        name: `${f_states}`,
        type: 'line',
        data: data,
        symbolSize: 7,
        animation: false
      })
    }
    option.series.sort((a: any, b: any) => -a.data[10] + b.data[10])
    chart.setOption(option)
  }
  const drawLine2 = (chartDom: HTMLElement) => {
    const K: number[] = []
    for (let i = 1; i <= 100; i++) {
      K.push(i)
    }
    const option: any = {
      tooltip: {
        trigger: 'axis'
      },
      toolbox: {
        feature: {
          saveAsImage: {
            pixelRatio: 8
          }
        }
      },

      legend: {
        width: '10%',
        top: 80,
        right: 60
      },
      grid: {
        top: '60px',
        left: '60px',
        right: '60px',
        bottom: '20px'
        // containLabel: true
      },
      yAxis: [
        {
          name: 'Discounted Sum',
          type: 'value',
          // max: 1,
          nameTextStyle: { fontSize: 14 }
        }
      ],
      xAxis: {
        name: 'K_max',
        type: 'category',
        data: K
      },
      series: []
    }

    const chart = echarts.init(chartDom)
    watch(
      gamma,
      () => {
        option.series = []
        for (const f_states of generatePermutations([0, 1, 2, 3, 4])) {
          const data = []

          for (const kk of K) {
            let discounted_sum = 0
            for (let k = 1; k <= kk; k++) {
              discounted_sum += Math.pow(gamma.value, k) * probs[JSON.stringify(f_states)][k]
            }
            data.push(discounted_sum)
          }
          option.series.push({
            name: `${f_states}`,
            type: 'line',
            data: data,
            symbolSize: 7,
            animation: false
          })
        }
        option.series.sort((a: any, b: any) => -a.data[10] + b.data[10])
        chart.setOption(option)
      },
      { immediate: true }
    )
  }

  const drawLine3 = (chartDom: HTMLElement) => {
    const K: number[] = []
    for (let i = 1; i <= 40; i++) {
      K.push(i)
    }
    const option: any = {
      tooltip: {
        trigger: 'axis'
      },
      toolbox: {
        feature: {
          saveAsImage: {
            pixelRatio: 8
          }
        }
      },

      legend: {
        width: '10%',
        top: 80,
        right: 60
      },
      grid: {
        top: '60px',
        left: '80px',
        right: '60px',
        bottom: '20px'
        // containLabel: true
      },
      yAxis: [
        {
          name: 'Transition Probability',
          type: 'value',
          // max: 1,
          nameTextStyle: { fontSize: 14 }
        }
      ],
      xAxis: {
        name: 'k-step',
        type: 'category',
        data: K
      },
      series: []
    }

    const chart = echarts.init(chartDom)

    for (let s = 0; s < n_state; s++) {
      option.series.push({
        type: 'line',
        data: [],
        animation: false
      })
    }
    for (const k of K) {
      const m = matrixPower(transitions, k)
      for (let s = 0; s < n_state; s++) {
        option.series[s].data.push(m[0][s])
      }
    }
    chart.setOption(option)
  }
  return {
    drawGraph,
    drawLine,
    drawLine2,
    drawLine3,
    k,
    gamma,
    kStepTransition,
    transitions,
    feasible_states,
    flexibility
  }
}

function generatePermutations(array: number[]): number[][] {
  const result: number[][] = []

  function backtrack(start: number, current: number[]) {
    result.push([...current])

    for (let i = start; i < array.length; i++) {
      current.push(array[i])
      backtrack(i + 1, current)
      current.pop()
    }
  }

  backtrack(1, [0])
  return result
}
