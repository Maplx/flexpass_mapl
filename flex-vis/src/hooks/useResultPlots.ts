import * as echarts from 'echarts'

interface Result {
  yAxis: Axis
  xAxis: Axis
  series: Series[]
}

interface Axis {
  name: string
  data: number[]
}

interface Series {
  name: string
  data: number[]
}

export function useResultPlots(result: Result, chartDom: HTMLElement) {
  const option: any = {
    grid: {
      top: '30px',
      left: '50px',
      right: '10px',
      bottom: '35px'
    },
    toolbox: {
      feature: {
        saveAsImage: {
          pixelRatio: 8
        }
      }
    },
    legend: <echarts.LegendComponentOption>{
      textStyle: {
        fontSize: 11,
        color: 'black'
      },
      left: 'right',
      itemGap: 4,
      right: 20,
      top: 3,
      itemWidth: 18,
      itemHeight: 10
    },
    xAxis: {
      name: result.xAxis.name,
      nameLocation: 'middle',
      nameGap: 22,
      type: 'category',
      data: result.xAxis.data,
      nameTextStyle: {
        color: 'black'
      },
      axisLabel: {
        color: 'black'
      }
    },
    yAxis: [
      {
        nameGap: 10,
        name: result.yAxis.name,
        type: result.yAxis.name == 'Time (s)' ? 'log' : 'value',
        nameTextStyle: {
          color: 'black'
        },
        axisLabel: {
          color: 'black'
        }
      }
    ],
    series: [
      {
        type: 'line',
        lineStyle: {
          width: 2
        },
        name: result.series[0].name,
        data: result.series[0].data
      },
      {
        type: 'line',
        lineStyle: {
          width: 2
        },
        name: result.series[1].name,
        data: result.series[1].data
      }
    ]
  }

  const chart = echarts.init(chartDom)
  chart.setOption(option)
}
