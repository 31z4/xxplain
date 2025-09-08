# xxplane

`EXPLAIN` на стероидах.
Это инструмент для разработчиков, аналитиков и администраторов баз данных, который:

&nbsp;🔮 Предсказывает время запроса до его выполнения.<br>
&nbsp;🔎 Наглядно представляет план запроса для более детального анализа.<br>
&nbsp;💡 Даёт рекомендации по оптимизации запроса.

Совместим с [PostgreSQL](https://www.postgresql.org) версии 15 и выше.

## Быстрый старт

Для работы потребуется:

* [Docker Compose](https://docs.docker.com/compose/)
* [uv](https://docs.astral.sh/uv/)

Следуй официальным инструкциям по установке.

### 1. Подготовка БД с синтетическими данными

Пропусти этот шаг, если хочешь подключить сервис к уже существующей БД.

1. Создай пароль суперпользователя и `xxplain` для Postgres:

    ```shell
    $ cat << EOF > .env.postgres
    POSTGRES_PASSWORD=super-secret
    POSTGRES_XXPLAIN_PASSWORD=top-secret
    EOF
    ```

2. Создай чистый Docker volume для Postgres и папку для данных:

        make clean-docker-volumes
        mkdir data

3. Запусти Postgres:

        docker compose up -d postgres

4. Заполни БД синтетическими данными:

        ./scripts/gendata.sh

### 2. Запуск приложения

1. Укажи данные для доступа приложения к БД и OpenAI совместимому API:

    ```shell
    $ cat << EOF > .env.backend
    POSTGRES_DSN=postgresql://xxplain:top-secret@postgres/postgres

    OPENAI_API_KEY=secret-api-key
    OPENAI_API_BASE_URL=https://openrouter.ai/api/v1
    OPENAI_API_MODEL=openai/gpt-oss-20b:free
    EOF
    ```

2. Запусти сервис приложения:

        docker compose up -d app

Web интерфейс приложения доступен на [http://localhost:8000](http://localhost:8000).

⚠️ **Внимание** ⚠️

В целях демонстрации используются обученные на синтетических данных модели предсказания метрик запроса.
Для лучшей точности предсказания требуется обучение на данных и запросах с целевой системы.

## API

Кроме Web интерфейса, приложение предоставляет простой HTTP API.

**Проанализировать запрос без выполнения**

    curl -X POST 'http://127.0.0.1:8000/explain' --data-binary 'select * from part'

Формат ответа:

* **plan**: план запроса
* **prediction**: предсказанные метрики запроса
    * **cost**: стоимость запроса полученная от планировщика Postgres
    * **cost_class**: класс стоимости запроса от 1 до 5 (выше – дороже)
    * **total_time_ms**: общее время выполнения запроса в мсек.
    * **total_time_class**: класс общего времени выполнения запроса (выше - дольше)
    * **data_read_bytes**: объем читаемых данных в байтах
    * **data_read_class**: класс объема читаемых данных в байтах (выше – больше)

**Выполнить и проанализировать запрос**

    curl -X POST 'http://127.0.0.1:8000/explain?analyze=1' --data-binary 'select * from part'

Формат ответа:

* **plan**: план запроса
* **prediction**: предсказанные метрики запроса
    * **cost**: стоимость запроса полученная от планировщика Postgres
    * **cost_class**: класс стоимости запроса от 1 до 5 (выше – дороже)
    * **total_time_ms**: общее время выполнения запроса в мсек.
    * **total_time_class**: класс общего времени выполнения запроса (выше - дольше)
    * **data_read_bytes**: объем читаемых данных в байтах
    * **data_read_class**: класс объема читаемых данных в байтах (выше – больше)
* **actual**: фактические метрики запроса
    * **total_time_ms**: общее время выполнения запроса в мсек.
    * **total_time_class**: класс общего времени выполнения запроса (выше - дольше)
    * **data_read_bytes**: объем читаемых данных в байтах
    * **data_read_class**: класс объема читаемых данных в байтах (выше – больше)

**Оптимизировать запрос**

    curl -X POST 'http://127.0.0.1:8000/optimize' --data-binary @benchmarks/tpc-h/queries/q05.sql

Формат ответа:

* **query**: оптимизированный запрос
* **plan**: план оптимизированного запроса
* **prediction**: предсказанные метрики оптимизированного запроса
    * **cost**: стоимость запроса полученная от планировщика Postgres
    * **cost_class**: класс стоимости запроса от 1 до 5 (выше – дороже)
    * **total_time_ms**: общее время выполнения запроса в мсек.
    * **total_time_class**: класс общего времени выполнения запроса (выше - дольше)
    * **data_read_bytes**: объем читаемых данных в байтах
    * **data_read_class**: класс объема читаемых данных в байтах (выше – больше)

## Обучение моделей

Теперь модели обучаются для предсказания двух метрик: времени выполнения запроса (time) и объема данных (size). Поддерживаются модели: Ridge, Lasso, RandomForest, GradientBoost, XGBoost и CatBoost.

1. Сбор данных

    Запустить

    ```
    scripts/a_collect_sql_plans.py
    ```

    оно соберёт EXPLAIN ANALYZE всё-что-можно-собрать в
    `train_query_plans.csv` из папки `benchmarks/tpc-h/generated` и `test_query_plans.csv` из
    папки `benchmarks/tpc-h/queries`. Процесс долгий, придётся подождать. Остальные быстрые.

2. Обогаить датасеты фичами

    ```
    python scripts/b_enrich_dataset.py datasets/train_query_plans.csv > train_dataset.csv
    ```

3. Обучить модельки

    ```
    python scripts/c_train_models.py datasets/train_dataset.csv
    ```

4. Загрузчик моделей можно проверить так

    ```
    python ml/loader.py benchmarks/tpc-h/queries/q15.sql
    ```

Для сравнения метрик моделей см. [Сравнительная таблица метрик](Cравнительная\ таблица\ метрик.md)

## Ссылки

**Инструменты**

* [pganalyze](https://pganalyze.com)
* [PostgreSQL Workload Analyzer](https://github.com/powa-team/powa)
* [pgMustard](https://www.pgmustard.com)
* [FlameExplain](https://flame-explain.com)
* [PEV2](https://github.com/dalibo/pev2)
* [SQLSolver: Proving Query Equivalence Using Linear Integer Arithmetic](https://github.com/SJTU-IPADS/SQLSolver)
* [QED, the Query Equivalence Decider](https://github.com/qed-solver)

**Бенчмарки и данные**

* [The CTU Prague Relational Learning Repository](https://relational.fel.cvut.cz)
* [DSB Benchmark](https://github.com/microsoft/dsb)
* [A Benchmark for Real-Time Analytics Applications](https://github.com/timescale/rtabench)
* [TPC-DS benchmark kit with some modifications/fixes](https://github.com/gregrahn/tpcds-kit)
* [Supplementary materials for SlabCity: Whole-Query Optimization using Program Synthesis](https://github.com/eidos06/SlabCity)

**Модели**

* [Zero-Shot Cost Estimation Models](https://github.com/DataManagementLab/zero-shot-cost-estimation)
* [LLM for Index Recommendation](https://github.com/XinxinZhao798/LLMIndexAdvisor)
* [R-Bot: An LLM-based Query Rewrite System](https://github.com/curtis-sun/LLM4Rewrite)
* [LLMOpt: Query Optimization utilizing Large Language Models](https://github.com/lucifer12346/LLMOpt)
