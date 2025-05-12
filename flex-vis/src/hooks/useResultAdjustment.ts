import * as echarts from 'echarts'

import res from '@/assets/results/adjustmentReconfigOnly.json'

export function useResultAdjustment(chartDom: HTMLElement) {
  const option = {
    grid: [
      {
        height: '160px',
        top: '25px',
        left: '45px',
        right: '45px'
      },
      { top: '218px', left: '45px', right: '45px', bottom: '10px' }
    ],
    toolbox: {
      feature: {
        saveAsImage: {
          pixelRatio: 8
        }
      }
    },
    legend: <echarts.LegendComponentOption>{
      textStyle: {
        fontSize: 9
      },
      // left: 'right',
      itemGap: 4,
      // right: 20,
      top: 3,
      itemWidth: 18,
      itemHeight: 9
    },
    xAxis: [
      {
        name: 'Adjustment Event',
        nameTextStyle: {
          fontSize: 10,
          color: 'black'
        },
        type: 'category',
        data: res.xAxis,
        nameLocation: 'center',
        nameGap: 20,
        axisLabel: {
          fontSize: 9,
          color: 'black'
        }
      },
      {
        show: false,
        name: 'Adjustment Event',
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
        name: 'Flexibility',
        type: 'value',
        nameGap: 25,
        nameLocation: 'center',
        axisLabel: {
          fontSize: 9,
          color: 'black'
        }
      },
      {
        nameTextStyle: {
          fontSize: 10,
          color: 'black'
        },
        name: 'Num of Adjusted Apps',
        type: 'value',
        nameGap: 25,
        nameLocation: 'center',
        axisLabel: {
          fontSize: 9,
          color: 'black'
        }
      },
      {
        nameTextStyle: {
          fontSize: 10,
          color: 'black'
        },
        name: 'Time (s)',
        type: 'value',
        nameGap: 25,
        // inverse: true,
        nameLocation: 'center',
        axisLabel: {
          fontSize: 9,
          color: 'black'
        },
        position: 'left',

        gridIndex: 1
      }
    ],
    series: [
      {
        name: 'Flexibility',
        type: 'line',
        data: res.flex
      },
      {
        name: 'Num of affected apps',
        type: 'bar',
        data: res.n_adjusted_apps,
        yAxisIndex: 1
      },
      {
        name: 'Running time',
        type: 'bar',
        data: res.time,
        yAxisIndex: 2,
        xAxisIndex: 1
      }
    ]
  }

  const chart = echarts.init(chartDom)
  chart.setOption(option)
}
