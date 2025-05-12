<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useResultPlots } from '@/hooks/useResultPlots'

import res_sr_apps from '@/assets/results/sr/n_apps.json'
import res_flex_apps from '@/assets/results/flex/n_apps.json'
import res_time_apps from '@/assets/results/time/n_apps.json'

import res_sr_states from '@/assets/results/sr/n_states.json'
import res_flex_states from '@/assets/results/flex/n_states.json'
import res_time_states from '@/assets/results/time/n_states.json'

import res_sr_flows from '@/assets/results/sr/n_flows.json'
import res_flex_flows from '@/assets/results/flex/n_flows.json'
import res_time_flows from '@/assets/results/time/n_flows.json'

import res_sr_links from '@/assets/results/sr/n_links.json'
import res_flex_links from '@/assets/results/flex/n_links.json'
import res_time_links from '@/assets/results/time/n_links.json'

import res_sr_slots from '@/assets/results/sr/n_slots.json'
import res_flex_slots from '@/assets/results/flex/n_slots.json'
import res_time_slots from '@/assets/results/time/n_slots.json'
import { useResultAdjustment } from '@/hooks/useResultAdjustment'
import { useResultAdjustmentScale } from '@/hooks/useResultAdjustmentScale'

const chartDomAdjustment = ref()
const chartDomAdjustmentScale = ref()
const chartDomSRApps = ref()
const chartDomFlexApps = ref()
const chartDomTimeApps = ref()

const chartDomSRStates = ref()
const chartDomFlexStates = ref()
const chartDomTimeStates = ref()

const chartDomSRFlows = ref()
const chartDomFlexFlows = ref()
const chartDomTimeFlows = ref()

const chartDomSRLinks = ref()
const chartDomFlexLinks = ref()
const chartDomTimeLinks = ref()

const chartDomSRSlots = ref()
const chartDomFlexSlots = ref()
const chartDomTimeSlots = ref()

onMounted(() => {
  useResultAdjustment(chartDomAdjustment.value)
  useResultAdjustmentScale(chartDomAdjustmentScale.value)
  useResultPlots(res_sr_apps, chartDomSRApps.value)
  useResultPlots(res_flex_apps, chartDomFlexApps.value)
  useResultPlots(res_time_apps, chartDomTimeApps.value)

  useResultPlots(res_sr_states, chartDomSRStates.value)
  useResultPlots(res_flex_states, chartDomFlexStates.value)
  useResultPlots(res_time_states, chartDomTimeStates.value)

  useResultPlots(res_sr_flows, chartDomSRFlows.value)
  useResultPlots(res_flex_flows, chartDomFlexFlows.value)
  useResultPlots(res_time_flows, chartDomTimeFlows.value)

  useResultPlots(res_sr_links, chartDomSRLinks.value)
  useResultPlots(res_flex_links, chartDomFlexLinks.value)
  useResultPlots(res_time_links, chartDomTimeLinks.value)

  useResultPlots(res_sr_slots, chartDomSRSlots.value)
  useResultPlots(res_flex_slots, chartDomFlexSlots.value)
  useResultPlots(res_time_slots, chartDomTimeSlots.value)
})
</script>

<template>
  <el-card>
    <div ref="chartDomAdjustment" class="chart-adj"></div>
    <div ref="chartDomAdjustmentScale" class="chart-adj-scale"></div>
    <el-row :gutter="40">
      <el-col :span="3">
        <div ref="chartDomSRApps" class="chart"></div>
        <div ref="chartDomFlexApps" class="chart"></div>
        <div ref="chartDomTimeApps" class="chart"></div>
      </el-col>
      <el-col :span="3">
        <div ref="chartDomSRStates" class="chart"></div>
        <div ref="chartDomFlexStates" class="chart"></div>
        <div ref="chartDomTimeStates" class="chart"></div>
      </el-col>
      <el-col :span="3">
        <div ref="chartDomSRFlows" class="chart"></div>
        <div ref="chartDomFlexFlows" class="chart"></div>
        <div ref="chartDomTimeFlows" class="chart"></div>
      </el-col>
      <el-col :span="3">
        <div ref="chartDomSRLinks" class="chart"></div>
        <div ref="chartDomFlexLinks" class="chart"></div>
        <div ref="chartDomTimeLinks" class="chart"></div>
      </el-col>
      <el-col :span="3">
        <div ref="chartDomSRSlots" class="chart"></div>
        <div ref="chartDomFlexSlots" class="chart"></div>
        <div ref="chartDomTimeSlots" class="chart"></div>
      </el-col>
    </el-row>
  </el-card>
</template>

<style scoped>
.chart {
  width: 210px;
  height: 150px;
}
.chart-adj {
  width: 360px;
  height: 300px;
}
.chart-adj-scale {
  width: 240px;
  height: 430px;
}
</style>
