import * as echarts from 'echarts'

export function useMarkovChain() {
  const option = {}
  const draw = (chartDom: HTMLElement) => {
    const chart = echarts.init(chartDom)
    chart.setOption(option)
  }

  return { draw }
}
