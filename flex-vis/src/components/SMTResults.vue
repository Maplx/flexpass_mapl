<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useSMTResults } from '@/hooks/useSMTResults'

const { res, drawPartition, drawSchedule } = useSMTResults()

const chartDomPartition = ref()
const chartDomSchedule = ref()
onMounted(() => {
  drawPartition(chartDomPartition.value)
  drawSchedule(chartDomSchedule.value)
})
const flows = ref<any[]>([])
for (const app of res.apps) {
  for (const state of app.states) {
    for (const flow of state.flows) {
      flows.value.push({
        a: app.id,
        s: state.id,
        f: flow.id,
        tx: flow.tx,
        p: flow.p
      })
    }
  }
}
</script>

<template>
  <el-card>
    <el-row>
      <el-col :span="6">
        <el-table table-layout="auto" :data="res.apps" style="width: 100%; color: black">
          <el-table-column type="expand">
            <template #default="props">
              <div v-for="(state, s) in props.row.states" :key="s">
                State {{ s }}
                <el-table
                  table-layout="auto"
                  :data="state.flows"
                  style="font-size: 0.8rem; color: black"
                >
                  <el-table-column prop="id" label="ID" />
                  <el-table-column prop="tx" label="TXs" />
                  <el-table-column prop="p" label="Period" />
                </el-table>
              </div>
            </template>
          </el-table-column>
          <el-table-column prop="id" label="App" />
          <el-table-column prop="n_states" label="N states" />
          <el-table-column prop="satisfied_states" label="Satisfied states" />
          <el-table-column prop="flexibility" label="Flex" />
        </el-table>
      </el-col>
      <el-col :span="18">
        <div ref="chartDomPartition" class="chart"></div>
        <div ref="chartDomSchedule" class="chart"></div>
      </el-col>
    </el-row>
  </el-card>
</template>

<style scoped>
.chart {
  width: 100%;
  height: 800px;
}
</style>
