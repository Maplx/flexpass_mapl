import res from '@/assets/smt/res-0.json'
import * as echarts from 'echarts'

export function useSMTResults() {
  const drawPartition = (chartDom: HTMLElement) => {
    const option = {
      grid: { left: '30px', right: '30px', top: '55px', bottom: '30px' },
      tooltip: {
        position: 'top'
      },
      legend: {
        width: '80%'
      },
      xAxis: {
        name: 'Slot',
        type: 'category',
        data: <number[]>[]
      },
      yAxis: {
        name: 'Links',
        type: 'category',
        data: res.links
      },
      visualMap: {
        show: false,
        orient: 'horizontal',
        type: 'piecewise',
        width: '30%',
        left: '70%',
        top: '0%',
        pieces: []
      },
      series: [
        {
          name: 'Partition',
          type: 'heatmap',
          data: [],
          label: {
            show: false
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
          },
          animation: false
        }
      ]
    }
    const colors = [
      '#5470c6',
      '#91cc75',
      '#fac858',
      '#ee6666',
      '#73c0de',
      '#3ba272',
      '#fc8452',
      '#9a60b4',
      '#ea7ccc'
    ]
    for (let t = 0; t < res.T; t++) option.xAxis.data.push(t)
    for (let i = 0; i < res.apps.length; i++) {
      option.visualMap.pieces.push({ value: i, color: colors[i], label: `App ${i}` })
      for (const r of res.partitions[i]) {
        for (const e of r[1]) {
          option.series[0].data.push([r[0], e, i])
        }
      }
      for (let s = 0; s < res.apps[i].n_states; s++) {
        const txs = []
        for (const tx in res.schedule) {
          if (i == parseInt(tx.split('_').at(1)!) && s == parseInt(tx.split('_').at(2)!)) {
            const e = parseInt(tx.split('_').at(-1)!)
            txs.push({
              name: tx,
              value: [res.schedule[tx], e, i]
            })
          }
        }
        option.series.push({
          name: `A${i}-S${s}`,
          type: 'heatmap',
          data: txs,
          color: colors[i],
          label: {
            show: true,
            formatter: (item) => {
              const [_, i, s, f, k, h, e] = item.name.split('_')
              return `s${s}-f${f}-k${k}-h${h}`
            }
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
          },
          animation: false
        })
      }
    }

    const chart = echarts.init(chartDom)
    chart.setOption(option)
  }
  const drawSchedule = (chartDom: HTMLElement) => {
    const option = {}
    const chart = echarts.init(chartDom)
    chart.setOption(option)
  }

  return { res, drawPartition, drawSchedule }
}
