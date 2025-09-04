from .pg import explain
from .predict import predict


def _llm(sql: str, plan: str) -> str:
    return sql  # TODO: Здесь будет настоящий вызов модели.


async def recommend(sql: str) -> dict:
    """Временная поддельная реализация рекомендаций."""
    plan = await explain(sql, analyze=False, format="TEXT")
    new_sql = _llm(sql, plan)

    new_plan = await explain(new_sql, analyze=False, format="JSON")
    prediction = predict(new_sql, new_plan)

    return {
        "query": new_sql,
        "plan": new_plan,
        "prediction": prediction,
    }
