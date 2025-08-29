# xxplane

`EXPLAIN` на стероидах.
Это инструмент для разработчиков, аналитиков и администраторов баз данных, который:

&nbsp;🔮 Предсказывает время запроса до его выполнения.<br>
&nbsp;🔎 Наглядно представляет план запроса для более детального анализа.<br>
&nbsp;💡 Даёт рекомендации по оптимизации запроса.

Совместим с [PostgreSQL](https://www.postgresql.org) версии 15 и выше.

## Подготовка окружения

Для работы потребуется [Docker Compose](https://docs.docker.com/compose/).

1. Создай пароль суперпользователя Postgres.

        echo 'POSTGRES_PASSWORD=secret' > .env.postgres

2. Создай чистый Docker volume для Postgres.

        make clean-docker-volumes

3. Все готово для запуска 🚀

        docker compose up
