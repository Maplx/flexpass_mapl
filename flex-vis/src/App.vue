<script setup lang="ts">
import MarkovChain from './components/MarkovChain.vue'
import SMTResults from './components/SMTResults.vue'
import HeuristicDemo from './components/HeuristicDemo.vue'
import ResultPlots from './components/ResultPlots.vue'
</script>

<template>
  <main>
    <el-row>
      <el-col :offset="1" :span="22">
        <MarkovChain v-if="false" />
        <SMTResults v-if="false" />
        <HeuristicDemo v-if="false" />
        <ResultPlots v-if="true" />
      </el-col>
    </el-row>
  </main>
</template>

<style scoped></style>
