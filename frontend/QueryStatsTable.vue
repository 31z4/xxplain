<template>
  <table class="table ">
    <thead>
      <tr>
        <th scope="col"></th>
        <th scope="col">Стоимость</th>
        <th scope="col">Время</th>
        <th scope="col">Данные</th>
      </tr>
    </thead>
    <tbody>
      <tr v-if="prediction">
        <th scope="row">Прогноз</th>
        <td>
          <span class="badge text-class-1" v-if="prediction?.cost_class === 0">оч. дешевый</span>
          <span class="badge text-class-2" v-if="prediction?.cost_class === 1">дешевый</span>
          <span class="badge text-class-3" v-if="prediction?.cost_class === 2">средний</span>
          <span class="badge text-class-4" v-if="prediction?.cost_class === 3">дорогой</span>
          <span class="badge text-class-5" v-if="prediction?.cost_class === 4">оч. дорогой</span>
          <p class="mb-0">{{ foramtCost(prediction?.cost) }}</p>
        </td>
        <td>
          <span class="badge text-class-1" v-if="prediction?.total_time_class === 0">оч. быстрый</span>
          <span class="badge text-class-2" v-if="prediction?.total_time_class === 1">быстрый</span>
          <span class="badge text-class-3" v-if="prediction?.total_time_class === 2">средний</span>
          <span class="badge text-class-4" v-if="prediction?.total_time_class === 3">долгий</span>
          <span class="badge text-class-5" v-if="prediction?.total_time_class === 4">оч. долгий</span>
          <p class="mb-0">{{ formatDuration(prediction?.total_time_ms) }}</p>
        </td>
        <td>
          <span class="badge text-class-1" v-if="prediction?.data_read_class === 0">оч. мало</span>
          <span class="badge text-class-2" v-if="prediction?.data_read_class === 1">мало</span>
          <span class="badge text-class-3" v-if="prediction?.data_read_class === 2">средне</span>
          <span class="badge text-class-4" v-if="prediction?.data_read_class === 3">много</span>
          <span class="badge text-class-5" v-if="prediction?.data_read_class === 4">оч. много</span>
          <p class="mb-0">{{ foramtSize(prediction?.data_read_bytes) }}</p>
        </td>
      </tr>
      <tr v-if="actual?.total_time_ms">
        <th scope="row">Факт</th>
        <td>
          &mdash;
        </td>
        <td>
          <span class="badge text-class-1" v-if="actual?.total_time_class === 0">оч. быстрый</span>
          <span class="badge text-class-2" v-if="actual?.total_time_class === 1">быстрый</span>
          <span class="badge text-class-3" v-if="actual?.total_time_class === 2">средний</span>
          <span class="badge text-class-4" v-if="actual?.total_time_class === 3">долгий</span>
          <span class="badge text-class-5" v-if="actual?.total_time_class === 4">оч. долгий</span>
          <p class="mb-0">{{ formatDuration(actual?.total_time_ms) }}</p>
        </td>
        <td>
          <span class="badge text-class-1" v-if="actual?.data_read_class === 0">оч. мало</span>
          <span class="badge text-class-2" v-if="actual?.data_read_class === 1">мало</span>
          <span class="badge text-class-3" v-if="actual?.data_read_class === 2">средне</span>
          <span class="badge text-class-4" v-if="actual?.data_read_class === 3">много</span>
          <span class="badge text-class-5" v-if="actual?.data_read_class === 4">оч. много</span>
          <p class="mb-0">{{ foramtSize(actual?.data_read_bytes) }}</p>

        </td>
      </tr>
    </tbody>
  </table>
</template>

<script setup>
// Определяем props
const props = defineProps({
  prediction: {
    type: Object,
    default: null,
  },
  actual: {
    type: Object,
    default: null,
  },
})

// Методы
const formatDuration = (ms) => {
  if (ms < 0) ms = -ms
  const time = {
    дн: Math.floor(ms / 86400000),
    ч: Math.floor(ms / 3600000) % 24,
    мин: Math.floor(ms / 60000) % 60,
    сек: Math.floor(ms / 1000) % 60,
    мс: Math.floor(ms) % 1000,
  }
  return Object.entries(time)
    .filter((val) => val[1] !== 0)
    .map((val) => val[1] + " " + val[0])
    .join(", ")
}
function foramtCost(bytes, dp = 1) {
  const thresh = 1000;
  if (Math.abs(bytes) < thresh) {
    return bytes + ' ';
  }
  const units = ['k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'];
  let u = -1;
  const r = 10 ** dp;
  do {
    bytes /= thresh;
    ++u;
  } while (Math.round(Math.abs(bytes) * r) / r >= thresh && u < units.length - 1);
  return bytes.toFixed(dp) + ' ' + units[u];
}
function foramtSize(bytes, si = false, dp = 1) {
  const thresh = si ? 1000 : 1024;
  if (Math.abs(bytes) < thresh) {
    return bytes + ' B';
  }
  const units = si
    ? ['kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
    : ['KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB'];
  let u = -1;
  const r = 10 ** dp;
  do {
    bytes /= thresh;
    ++u;
  } while (Math.round(Math.abs(bytes) * r) / r >= thresh && u < units.length - 1);
  return bytes.toFixed(dp) + ' ' + units[u];
}
</script>
