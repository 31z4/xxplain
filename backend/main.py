from contextlib import asynccontextmanager

import psycopg
import uvicorn
from fastapi import FastAPI, Request

from .config import settings

app = FastAPI(debug=True)


def valid_explain_sql(input: str, analyze: bool) -> str:
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
    return f"EXPLAIN ({analyze}BUFFERS, SETTINGS, FORMAT JSON) {input}"


@asynccontextmanager
async def rollback():
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


@app.post("/explain")
async def post_explain(request: Request, analyze: bool = False):
    body = await request.body()
    sql = valid_explain_sql(body.decode(), analyze)

    async with rollback() as cur:
        await cur.execute(sql)
        obj = await cur.fetchone()

    return obj


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        reload_dirs="backend",
        reload=True,
    )
