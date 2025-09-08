<template>
  <table class="table">
    <thead>
      <tr>
        <th scope="col"></th>
        <th scope="col">Cost</th>
        <th scope="col">Total time</th>
        <th scope="col">Data read</th>
      </tr>
    </thead>
    <tbody>
      <tr v-if="prediction">
        <th scope="row">Прогноз</th>
        <td>
          <span
            :class="{
              'badge text-bg-success': prediction?.cost_class === 0,
              'badge text-bg-warning': prediction?.cost_class === 1,
              'badge text-bg-danger':
                prediction?.cost_class === 2 || prediction?.cost_class === 3,
            }"
            >{{ prediction?.cost }}</span
          >
        </td>
        <td>
          <span
            :class="{
              'badge text-bg-success': prediction?.total_time_class === 0,
              'badge text-bg-warning': prediction?.total_time_class === 1,
              'badge text-bg-danger': prediction?.total_time_class === 2,
            }"
          >
            {{ formatDuration(prediction?.total_time_ms) }}
          </span>
        </td>
        <td>
          <span
            :class="{
              'badge text-bg-success': prediction?.data_read_class === 0,
              'badge text-bg-warning': prediction?.data_read_class === 1,
              'badge text-bg-danger': prediction?.data_read_class === 2,
            }"
          >
            {{ foramtSize(prediction?.data_read_bytes) }}
          </span>
        </td>
      </tr>
      <tr v-if="actual?.total_time_ms">
        <th scope="row">Факт</th>
        <td>
          &mdash;
        </td>
        <td>
          <span
            :class="{
              'badge text-bg-success': actual?.total_time_class === 0,
              'badge text-bg-warning': actual?.total_time_class === 1,
              'badge text-bg-danger': actual?.total_time_class === 2,
            }"
          >
            {{ formatDuration(actual?.total_time_ms) }}
          </span>
        </td>
        <td>
          <span
            :class="{
              'badge text-bg-success': actual?.data_read_class === 0,
              'badge text-bg-warning': actual?.data_read_class === 1,
              'badge text-bg-danger': actual?.data_read_class === 2,
            }"
            >{{ foramtSize(actual?.data_read_bytes) }}</span
          >
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
