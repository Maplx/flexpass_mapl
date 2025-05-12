<script setup lang="tsx">
import { useHeuristic } from '@/hooks/useHeursitic'

const {
  E,
  T,
  apps,
  flexibility,
  all_feasible_states,
  new_feasible_states,
  partition,
  step,
  pass,
  revoke,
  black_list,
  cur_slot,
  cur_link,
  cur_gains,
  progress
} = useHeuristic()

const generateColumnsPartition = (length = E, props?: any) =>
  Array.from({ length }).map((_, columnIndex) => ({
    ...props,
    key: `slots[${columnIndex}]`,
    dataKey: `slots[${columnIndex}]`,
    title: `Slot ${columnIndex}`,
    width: 90,
    cellRenderer: ({ cellData: v }: any) => {
      if (v.app == -1) return ''

      if (!v.is_checkpoint) return `a${v.app},s[${v.states}]`

      return (
        <span style="color:red;font-weight:600">
          a{v.app},s[{`${v.states}`}]
        </span>
      )
    }
  }))

const columnsPartition = generateColumnsPartition(T)
columnsPartition.unshift({
  key: 'link',
  dataKey: 'link',
  title: `Link`,
  width: 65
})

const generateColumnsFlows = (length = 10, props?: any) =>
  Array.from({ length }).map((_, columnIndex) => ({
    ...props,
    key: `states`,
    dataKey: `states[${columnIndex}]`,
    title: `State ${columnIndex}`,
    width: 75,
    cellRenderer: ({ cellData: v }: any) => (v != undefined ? JSON.stringify(v.flows) : '')
  }))

const columnsFlows = generateColumnsFlows(5)
columnsFlows.unshift({
  key: 'id',
  dataKey: 'id',
  title: `App`,
  width: 65
})

const generateColumnsProgress = (length = 10, props?: any) =>
  Array.from({ length }).map((_, columnIndex) => ({
    ...props,
    key: `states[${columnIndex}]`,
    dataKey: `states[${columnIndex}]`,
    title: `State ${columnIndex}`,
    width: 75,
    cellRenderer: ({ cellData: v }: any) => {
      if (v != undefined) {
        if (v.cur != v.total) return `${v.cur}/${v.total}`
        return (
          <span style="color:royalblue;font-weight:600">
            {v.cur}/{v.total}
          </span>
        )
      }
      return ''
    }
  }))

const columnsProgress = generateColumnsProgress(5)
columnsProgress.unshift({
  key: 'app',
  dataKey: 'app',
  title: `App`,
  width: 65
})
</script>

<template>
  <el-card>
    <el-row>
      <el-col :span="12">
        Allocate link {{ cur_link }} in slot {{ cur_slot }}
        <el-button @click="step" size="small">Step</el-button>
        <el-button @click="pass" size="small">Pass</el-button>
        <el-button @click="revoke" size="small">Revoke</el-button>
        <br />
        Potential gains: {{ cur_gains }} <br /><br />
        New feasible states: {{ new_feasible_states }}
        <br />
        All feasible states: {{ all_feasible_states }}
        <br />
        Flexibility: {{ flexibility }}

        <br />
        Black list: {{ black_list }}
        <br />
        <!-- Check points: {{ check_points }} -->
      </el-col>
      <el-col :span="3">
        <!-- Flows
        <el-table-v2
          :columns="columnsFlows"
          :data="apps"
          :width="450"
          :height="450"
          :estimated-row-height="40"
          fixed
        /> -->
      </el-col>
      <el-col :offset="1" :span="8">
        Feasibility progress
        <el-table-v2
          :columns="columnsProgress"
          :data="progress"
          :width="450"
          :height="250"
          :row-height="40"
          fixed
        />
      </el-col>
    </el-row>

    <el-table-v2
      :columns="columnsPartition"
      :data="partition"
      :width="2000"
      :height="650"
      :row-height="40"
      fixed
    />
    <br />
    {{ apps }}
  </el-card>
</template>

<style scoped>
.chart {
  width: 400px;
  height: 400px;
}
</style>
