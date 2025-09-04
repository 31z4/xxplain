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

## Обучение моделей

```shell
uv run scripts/build_datasets.py
uv run scripts/train_models.py
```

Файлы соберутся в `datasets` и `models` соответственно.
Проверить работу моделей можно с помощью скрипта к бекенду:

```shell
uv run scripts/api_usage.py
```

## Фронтенд

Запускаю так:

```shell
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

В целом всё бесхитрострно лежит в `index.html` и написано на Vue (из-за готового PEV2)


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

**Модели**

* [Zero-Shot Cost Estimation Models](https://github.com/DataManagementLab/zero-shot-cost-estimation)
* [LLM for Index Recommendation](https://github.com/XinxinZhao798/LLMIndexAdvisor)
* [R-Bot: An LLM-based Query Rewrite System](https://github.com/curtis-sun/LLM4Rewrite)
* [LLMOpt: Query Optimization utilizing Large Language Models](https://github.com/lucifer12346/LLMOpt)
