import csv
import io
from pathlib import Path
from typing import Iterable

import psycopg
import sqlparse
import structlog
from openai import AsyncOpenAI
from psycopg.rows import dict_row
from sqlglot import exp, parse

from .config import settings
from .pg import explain
from .predict import predict

_SYSTEM_PROMPT_TEMPLATE = (
    "You are a Postgres 15 database expert and SQL optimizer. "
    "Your role involves identifying inefficient SQL queries and transforming "
    "them into optimized semantically equivalent and syntactically correct versions. "
    "Make sure an optimized query has the same output and doesn't modify the "
    "predicates even if it seems wrong. "
    "Only output the optimized query in one line, don't include any other "
    "additional words or newline characters. "
    "You are given the following Postgres 15 database schema to help you with "
    "rewriting the queries:\n"
    "\n{schema}\n"
    "You are also given the following table stats in CSV format to help you with "
    "rewriting the queries:\n"
    "\n{stats}\n"
    "A user has provided the following query that is potentially inefficient:"
)

_client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
    base_url=str(settings.OPENAI_API_BASE_URL),
)

log = structlog.stdlib.get_logger()


def _tables(schema: str) -> Iterable[str]:
    tables = set()
    for e in parse(schema):
        tables.update(t.name for t in e.find_all(exp.Table))
    return tables


def _csv(input: list[dict]) -> str:
    output = io.StringIO()
    fieldnames = input[0].keys()
    writer = csv.DictWriter(output, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(input)

    return output.getvalue()


async def table_stats(tables: Iterable[str]) -> str:
    async with await psycopg.AsyncConnection.connect(
        str(settings.POSTGRES_DSN), row_factory=dict_row
    ) as conn:
        cur = await conn.execute(
            """
            SELECT
                relname AS tablename,
                n_live_tup AS row_count,
                pg_size_pretty(pg_table_size(relid)) AS table_size_excluding_indexes
            FROM pg_stat_user_tables
            WHERE relname = ANY(%s)
        """,
            [list(tables)],
        )
        objs = await cur.fetchall()

    return objs


async def llm(sql: str) -> str:
    schema = Path(settings.POSTGRES_DB_SCHEMA).read_text(encoding="utf-8")
    tables = _tables(schema)
    stats = await table_stats(tables)
    stats = _csv(stats)

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(schema=schema, stats=stats)

    log.debug("Запрос к LLM", system_prompt=system_prompt, user_prompt=sql)
    completion = await _client.chat.completions.create(
        model=settings.OPENAI_API_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sql},
        ],
    )
    log.debug("Ответ LLM", completion=completion)

    return completion.choices[0].message.content


async def optimize(sql: str) -> dict:
    """Рекомендует потенциально оптимизированную версию sql запроса."""
    if not settings.OPENAI_API_KEY:
        log.warning(
            "Для получения рекомендаций установи переменную среды OPENAI_API_KEY"
        )
        return {}

    old_plan = await explain(sql, analyze=False, format="JSON")
    old_prediction = predict(sql, old_plan)

    try:
        new_sql = await llm(sql)
    except BaseException:
        log.exception("Не удалось получить рекомендации")
        return {}

    try:
        new_plan = await explain(new_sql, analyze=False, format="JSON")
    except (
        psycopg.errors.ProgrammingError,
        psycopg.errors.NotSupportedError,
        psycopg.errors.DataError,
    ):
        log.exception("Не валидный запрос в рекомендации")
        return {}

    # Хак: если метрики не изменились, значит с высокой долей вероятности наша
    # рекомендация такая же как и оригинальный запрос.
    # Для прода, тут лучше проверять семантическую эквивалентность запроса.
    # Метод из sqlglot (https://sqlglot.com/sqlglot/diff.html) тут работает плохо.
    # Надо пробовать альтернативные методы, например https://github.com/qed-solver
    prediction = predict(new_sql, new_plan)
    if prediction == old_prediction:
        log.warning("Метрики рекомендованного запроса не изменились")
        return {}

    new_sql = sqlparse.format(new_sql, reindent_aligned=True, indent_width=4)

    return {
        "query": new_sql,
        "plan": new_plan,
        "prediction": prediction,
    }
