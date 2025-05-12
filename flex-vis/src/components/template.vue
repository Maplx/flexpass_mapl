<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useMarkovChain } from '@/hooks/useMarkovChain'

const { draw } = useMarkovChain()
const chartDom = ref()
onMounted(() => {
  draw(chartDom.value)
})
</script>

<template>
  <el-card>
    <div ref="chartDom" class="chart"></div>
  </el-card>
</template>

<style scoped>
.chart {
  width: 400px;
  height: 400px;
}
</style>
