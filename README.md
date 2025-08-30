# xxplane

`EXPLAIN` на стероидах.
Это инструмент для разработчиков, аналитиков и администраторов баз данных, который:

&nbsp;🔮 Предсказывает время запроса до его выполнения.<br>
&nbsp;🔎 Наглядно представляет план запроса для более детального анализа.<br>
&nbsp;💡 Даёт рекомендации по оптимизации запроса.

Совместим с [PostgreSQL](https://www.postgresql.org) версии 15 и выше.

## Подготовка окружения

Для работы потребуется:

* [Docker Compose](https://docs.docker.com/compose/)
* [uv](https://docs.astral.sh/uv/)

Следуй официальным инструкциям по установке.

1. Создай пароль суперпользователя и `xxplain` для Postgres.

    ```shell
    $ cat << EOF > .env.postgres
    POSTGRES_PASSWORD=super-secret
    POSTGRES_XXPLAIN_PASSWORD=top-secret
    EOF
    ```

2. Укажи, как сервис будет подключаться к Postgres:

        echo 'POSTGRES_DSN=postgresql://xxplain:top-secret@localhost/postgres' > .env.backend

3. Создай чистый Docker volume для Postgres.

        make clean-docker-volumes

4. Все готово для запуска 🚀

        docker compose up
        make backend-run

### Подготовка синтетических данных

Имея чистый Docker volume и запущенный Postgres, выполни скрипт:

    ./scripts/gendata.sh

## Сбор данных для обучения моделей

    uv run lab/build_datasets.py

Файлы соберутся в `lab/train.csv` и `lab/test.csv`. В репозиторий положил в zip

Проверка работы разных моделей на фичах, собранных в train.csv - в файле `predict_model.py`.
Запуск сравнения моделей:

    uv run lab/predict_model.py

Предсказания будут сохранениы в `lab/test_with_pred_{Model Name}.csv`
На момент написания из всех моделей лучше всего показала себя GradientBoosting.

## Ссылки

**Инструменты**

* [pganalyze](https://pganalyze.com)
* [PostgreSQL Workload Analyzer](https://github.com/powa-team/powa)
* [pgMustard](https://www.pgmustard.com)
* [FlameExplain](https://flame-explain.com)

**Бенчмарки и данные**

* [The CTU Prague Relational Learning Repository](https://relational.fel.cvut.cz)
* [DSB Benchmark](https://github.com/microsoft/dsb)
* [A Benchmark for Real-Time Analytics Applications](https://github.com/timescale/rtabench)
* [TPC-DS benchmark kit with some modifications/fixes](https://github.com/gregrahn/tpcds-kit)

**Модели**

* [Zero-Shot Cost Estimation Models](https://github.com/DataManagementLab/zero-shot-cost-estimation)
* [LLM for Index Recommendation](https://github.com/XinxinZhao798/LLMIndexAdvisor)
