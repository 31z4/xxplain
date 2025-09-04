<template>
  <div class="card mb-3">
    <div class="border-bottom d-flex">
      <button type="button" class="btn">{{ title }}</button>
      <div class="flex-fill"></div>
      <button
        type="button"
        ref="tooltip1"
        class="btn btn-clipboard"
        data-bs-toggle="tooltip"
        data-bs-placement="top"
        data-bs-title="Скопировать"
        @click="handleCopyToClipboard"
      >
        <div v-if="!copied">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            fill="currentColor"
            class="bi bi-clipboard"
            viewBox="0 0 16 16"
          >
            <path
              d="M4 1.5H3a2 2 0 0 0-2 2V14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3.5a2 2 0 0 0-2-2h-1v1h1a1 1 0 0 1 1 1V14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1h1z"
            />
            <path
              d="M9.5 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5zm-3-1A1.5 1.5 0 0 0 5 1.5v1A1.5 1.5 0 0 0 6.5 4h3A1.5 1.5 0 0 0 11 2.5v-1A1.5 1.5 0 0 0 9.5 0z"
            />
          </svg>
        </div>
        <div v-if="copied">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            fill="currentColor"
            class="bi bi-check"
            viewBox="0 0 16 16"
          >
            <path
              d="M10.97 4.97a.75.75 0 0 1 1.07 1.05l-3.99 4.99a.75.75 0 0 1-1.08.02L4.324 8.384a.75.75 0 1 1 1.06-1.06l2.094 2.093 3.473-4.425z"
            />
          </svg>
        </div>
      </button>
      <button
        type="button"
        rel="tooltip2"
        class="btn btn-clipboard"
        data-bs-toggle="tooltip"
        data-bs-placement="top"
        data-bs-title="Применить"
        @click="handleApplySQL"
      >
        <div v-if="!applied">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            fill="currentColor"
            class="bi bi-stars"
            viewBox="0 0 16 16"
          >
            <path
              d="M7.657 6.247c.11-.33.576-.33.686 0l.645 1.937a2.89 2.89 0 0 0 1.829 1.828l1.936.645c.33.11.33.576 0 .686l-1.937.645a2.89 2.89 0 0 0-1.828 1.829l-.645 1.936a.361.361 0 0 1-.686 0l-.645-1.937a2.89 2.89 0 0 0-1.828-1.828l-1.937-.645a.361.361 0 0 1 0-.686l1.937-.645a2.89 2.89 0 0 0 1.828-1.828zM3.794 1.148a.217.217 0 0 1 .412 0l.387 1.162c.173.518.579.924 1.097 1.097l1.162.387a.217.217 0 0 1 0 .412l-1.162.387A1.73 1.73 0 0 0 4.593 5.69l-.387 1.162a.217.217 0 0 1-.412 0L3.407 5.69A1.73 1.73 0 0 0 2.31 4.593l-1.162-.387a.217.217 0 0 1 0-.412l1.162-.387A1.73 1.73 0 0 0 3.407 2.31zM10.863.099a.145.145 0 0 1 .274 0l.258.774c.115.346.386.617.732.732l.774.258a.145.145 0 0 1 0 .274l-.774.258a1.16 1.16 0 0 0-.732.732l-.258.774a.145.145 0 0 1-.274 0l-.258-.774a1.16 1.16 0 0 0-.732-.732L9.1 2.137a.145.145 0 0 1 0-.274l.774-.258c.346-.115.617-.386.732-.732z"
            />
          </svg>
        </div>
        <div v-if="applied">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            fill="currentColor"
            class="bi bi-check"
            viewBox="0 0 16 16"
          >
            <path
              d="M10.97 4.97a.75.75 0 0 1 1.07 1.05l-3.99 4.99a.75.75 0 0 1-1.08.02L4.324 8.384a.75.75 0 1 1 1.06-1.06l2.094 2.093 3.473-4.425z"
            />
          </svg>
        </div>
      </button>
    </div>
    <div class="card-body">
      <code>{{ sql }}</code>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

// Определяем props
const props = defineProps({
  sql: {
    type: String,
    required: true,
  },
  title: {
    type: String,
    required: false,
  },
})

// Определяем события
const emit = defineEmits(['apply-sql'])

// Реактивные данные
const copied = ref(false)
const applied = ref(false)

// Инициализация Bootstrap tooltips при монтировании компонента
onMounted(() => {
  const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]')
  const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl))
})

// Методы
const handleCopyToClipboard = () => {
  copied.value = true
  navigator.clipboard.writeText(props.sql).then(() => {
    // Можно добавить визуальную обратную связь, например, изменить значок кнопки или показать уведомление
  })
  setTimeout(() => {
    copied.value = false
  }, 1000)
}

const handleApplySQL = () => {
  applied.value = true
  emit('apply-sql', props.sql)
  console.log('emit applySql')

  // Сбросить состояние через 1 секунду
  setTimeout(() => {
    applied.value = false
  }, 1000)
}
</script>
