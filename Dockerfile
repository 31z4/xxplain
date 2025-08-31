FROM python:3.13-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Создание пользователя приложения
RUN useradd --create-home --shell /bin/bash app

# Установка рабочей директории
WORKDIR /app

# Копирование файлов зависимостей
COPY pyproject.toml ./

# Установка Python зависимостей
RUN pip install --no-cache-dir -e .

# Копирование исходного кода
COPY ml/ ./ml/
COPY backend/ ./backend/
COPY models/ ./models/

# Изменение владельца файлов
RUN chown -R app:app /app

# Переключение на пользователя приложения
USER app

# Экспорт порта
EXPOSE 8000

# Команда запуска
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]