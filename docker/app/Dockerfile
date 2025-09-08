FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV \
    # Включить компиляцию байткода для ускорения запуска.
    UV_COMPILE_BYTECODE=1 \
    # Копировать из кэша вместо линковки, так как это смонтированный том.
    UV_LINK_MODE=copy

WORKDIR /app

# Установить зависимости проекта, используя lockfile и настройки.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

COPY ml /app/ml
COPY time_models /app/time_models
COPY size_models /app/size_models
COPY frontend /app/frontend
COPY backend /app/backend
COPY benchmarks/tpc-h/schema.sql /app/benchmarks/tpc-h/schema.sql

FROM python:3.13-slim-bookworm

WORKDIR /app
USER nobody

COPY --from=builder --chown=nobody:nobody /app /app

# Поместить исполняемые файлы окружения в начало PATH.
ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "-m", "backend.main"]
