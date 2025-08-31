from contextlib import asynccontextmanager

import psycopg
import uvicorn
from fastapi import FastAPI, Request, Depends, HTTPException

from .config import settings
from .predict import predict
from .models import (
    PredictionRequest, PredictionResponse
)
from .services import get_prediction_service, PredictionService

app = FastAPI(
    title="xxplain - Query Performance Prediction API",
    description="API для предсказания времени выполнения SQL запросов в PostgreSQL",
    version="0.1.0",
    debug=True
)


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

    analyze_clause = "ANALYZE, " if analyze else ""
    return f"EXPLAIN ({analyze_clause}BUFFERS, SETTINGS, FORMAT JSON) {input}"


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
    """Получает query plan для SQL запроса"""
    body = await request.body()
    sql = valid_explain_sql(body.decode(), analyze)

    async with rollback() as cur:
        await cur.execute(sql)  # type: ignore
        obj = await cur.fetchone()

    plan = obj[0][0]
    prediction = predict(plan)

    return {
        "plan": plan,
        "prediction": prediction,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_execution_time(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service)
) -> PredictionResponse:
    """
    Предсказывает время выполнения запроса по query plan
    """
    try:
        result = await service.predict_regression(
            query_text=request.query_text,
            model_name=request.model_name,
        )
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        reload_dirs="backend",
        reload=True,
    )
