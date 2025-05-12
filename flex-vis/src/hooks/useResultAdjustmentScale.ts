import * as echarts from 'echarts'

import res from '@/assets/results/dynamic/n_apps.json'

export function useResultAdjustmentScale(chartDom: HTMLElement) {
  const option = {
    grid: [
      {
        top: '25px',
        height: '160px',
        left: '45px',
        right: '5px'
      },
      {
        top: '225px',
        height: '160px',
        left: '45px',
        right: '5px'
      }
    ],
    toolbox: {
      feature: {
        saveAsImage: {
          pixelRatio: 8
        }
      }
    },
    xAxis: [
      {
        name: res.name,
        nameTextStyle: {
          fontSize: 10,
          color: 'black'
        },
        type: 'category',
        data: res.xAxis,
        nameLocation: 'middle',
        nameGap: 22,
        axisLabel: {
          fontSize: 9,
          color: 'black'
        }
      },
      {
        name: res.name,
        nameTextStyle: {
          fontSize: 10,
          color: 'black'
        },
        type: 'category',
        data: res.xAxis,
        nameLocation: 'middle',
        nameGap: 22,
        axisLabel: {
          fontSize: 9,
          color: 'black'
        },
        gridIndex: 1
      }
    ],
    yAxis: [
      {
        nameTextStyle: {
          fontSize: 10,
          color: 'black'
        },
        name: 'AO (%)',
        type: 'value',
        nameGap: 25,
        nameLocation: 'center',
        axisLabel: {
          fontSize: 9,
          color: 'black'
        },
        min: 0,
        max: 60
      },
      {
        nameTextStyle: {
          fontSize: 10,
          color: 'black'
        },
        min: 0,
        max: 60,
        name: 'TO (%)',
        type: 'value',
        nameGap: 25,
        nameLocation: 'center',
        axisLabel: {
          fontSize: 9,
          color: 'black'
        },
        gridIndex: 1
      }
    ],
    series: [
      {
        name: 'App',
        type: 'bar',
        barWidth: 20,
        data: res.app
      },
      {
        name: 'Time',
        type: 'bar',
        barWidth: 20,
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: res.time
      }
    ]
  }

  const chart = echarts.init(chartDom)
  chart.setOption(option)
}
