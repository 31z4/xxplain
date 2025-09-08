import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from .analyze import analyze_plan
from .config import settings
from .optimize import optimize
from .pg import explain
from .predict import predict

app = FastAPI(debug=settings.DEBUG)


@app.post("/explain")
async def post_explain(request: Request, analyze: bool = False):
    """Получает query plan и прогнозирует метрики для SQL запроса."""
    body = await request.body()
    sql = body.decode()

    plan = await explain(sql, analyze, format="JSON")
    prediction = predict(sql, plan)
    analysis = analyze_plan(plan)

    return {
        "plan": plan,
        "prediction": prediction,
        "actual": analysis,
    }


@app.post("/optimize")
async def post_optimize(request: Request):
    """Рекомендует потенциально оптимизированную версию sql запроса."""
    body = await request.body()
    sql = body.decode()

    recommendation = await optimize(sql)
    return recommendation


# Только для разработки.
# В проде статику должен отдавать отдельный веб сервер (nginx, Caddy).
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",  # Для работы в Docker.
        reload_dirs="backend",
        reload=settings.DEBUG,
    )
