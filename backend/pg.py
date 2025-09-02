from contextlib import asynccontextmanager

import psycopg

from .config import settings


def _valid_explain_sql(input: str, analyze: bool, format: str) -> str:
    """Простейшая функция, которая защищает от нескольких statements.

    Внимание! Эта функция не закрывает все возможные вектора атаки.
    Для прода требуется тщательный аудит и защита на разных уровнях (БД, connection pooler).
    """
    # Можно иметь ; в конце.
    input = input.strip().rstrip(";")
    # Но нельзя в середине.
    if ";" in input:
        raise ValueError

    analyze = "ANALYZE, " if analyze else ""
    return f"EXPLAIN ({analyze}BUFFERS, SETTINGS, FORMAT {format}) {input}"


@asynccontextmanager
async def _rollback():
    """Создает соединение и транзакцию, которая никогда не будет закомичена."""
    conn = await psycopg.AsyncConnection.connect(str(settings.POSTGRES_DSN))
    try:
        async with conn.cursor() as cur:
            yield cur
    finally:
        # Тут отсутствует явный вызов conn.commit(), так что наша транзакция будет
        # отменена, как только соединение будет закрыто.
        # Подробнее: https://www.psycopg.org/psycopg3/docs/basic/transactions.html
        await conn.close()


async def explain(sql: str, analyze: bool, format: str) -> dict | str:
    """Выполняет EXPLAIN для данного sql и возвращает результат в заданном формате."""
    explain_sql = _valid_explain_sql(sql, analyze, format)

    async with _rollback() as cur:
        await cur.execute(explain_sql)
        objs = await cur.fetchall()

    fmt = format.lower()
    if fmt == "json":
        return objs[0][0][0]
    if fmt == "text":
        return "\n".join(i[0] for i in objs)
    raise ValueError
