from pathlib import Path

import structlog
from openai import AsyncOpenAI

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
    "A user has provided the following query that is potentially inefficient:"
)

_client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
    base_url=str(settings.OPENAI_API_BASE_URL),
)

log = structlog.stdlib.get_logger()


async def llm(sql: str) -> str:
    schema = Path(settings.POSTGRES_DB_SCHEMA).read_text(encoding="utf-8")
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(schema=schema)

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

    try:
        new_sql = await llm(sql)
    except BaseException:
        log.exception("Не удалось получить рекомендации")
        return {}

    new_plan = await explain(new_sql, analyze=False, format="JSON")
    prediction = predict(new_sql, new_plan)

    return {
        "query": new_sql,
        "plan": new_plan,
        "prediction": prediction,
    }
