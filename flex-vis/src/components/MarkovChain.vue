<script setup lang="tsx">
import { ref, onMounted, watch } from 'vue'
import { useMarkovChain } from '@/hooks/useMarkovChain'

const {
  drawGraph,
  drawLine,
  drawLine2,
  drawLine3,
  k,
  gamma,
  kStepTransition,
  transitions,
  flexibility,
  feasible_states
} = useMarkovChain()

const chartDomGraph = ref()
const chartDomLine = ref()
const chartDomLine2 = ref()
const chartDomLine3 = ref()
onMounted(() => {
  drawGraph(chartDomGraph.value)
  drawLine(chartDomLine.value)
  drawLine2(chartDomLine2.value)
  drawLine3(chartDomLine3.value)
})
const transitions_display = JSON.parse(JSON.stringify(transitions))
const transitions_display_k_step = ref(JSON.parse(JSON.stringify(transitions)))

const columns_transition = Array.from({ length: transitions.length }).map((_, i) => ({
  key: `${i + 1}`,
  dataKey: `${i + 1}`,
  title: `State ${i}`,
  width: 100,
  align: 'center'
}))
columns_transition.unshift({
  key: '0',
  dataKey: '0',
  title: '',
  width: 70,
  align: 'center'
})
for (let i = 0; i < transitions.length; i++) {
  transitions_display[i].unshift(`State ${i}`)
  transitions_display_k_step.value[i].unshift(`State ${i}`)
}
watch(kStepTransition, () => {
  transitions_display_k_step.value = JSON.parse(JSON.stringify(kStepTransition.value))
  for (let i = 0; i < transitions.length; i++) {
    transitions_display_k_step.value[i].unshift(`State ${i}`)
  }
})
</script>

<template>
  <el-card>
    Feasible states:{{ feasible_states }}<br />
    Flexibility: {{ flexibility.toFixed(6) }} <br />
    <el-row :gutter="40">
      <el-col :span="12">
        <div ref="chartDomGraph" class="chart"></div>
        <el-table-v2
          :columns="columns_transition"
          :data="transitions_display"
          :width="600"
          :height="400"
          fixed
        />
        <div ref="chartDomLine3" class="chart"></div>
      </el-col>
      <el-col :span="12">
        <div ref="chartDomLine" class="chart" style="height: 500px"></div>
        <div ref="chartDomLine2" class="chart" style="height: 500px"></div>
        k step: <el-input-number v-model="k" :min="1" :max="100" />

        gamma:<el-input-number v-model="gamma" :precision="2" :step="0.02" :min="0.1" :max="1" />
        <el-table-v2
          :columns="columns_transition"
          :data="transitions_display_k_step"
          :width="600"
          :height="400"
          fixed
        />
      </el-col>
    </el-row>
  </el-card>
</template>

<style>
.chart {
  width: 100%;
  height: 400px;
  margin-bottom: 30px;
}
</style>
