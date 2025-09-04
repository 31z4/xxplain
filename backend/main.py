import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from .analyze import analyze_plan
from .pg import explain
from .predict import predict
from .recommend import recommend

app = FastAPI(debug=True)


@app.post("/explain")
async def post_explain(request: Request, analyze: bool = False):
    """Получает query plan для SQL запроса"""
    body = await request.body()
    sql = body.decode()

    plan = await explain(sql, analyze, format="JSON")
    prediction = predict(sql, plan)
    analysis = analyze_plan(plan)
    recommendation = await recommend(sql)

    return {
        "plan": plan,
        "prediction": prediction,
        "actual": analysis,
        "recommendation": recommendation,
    }


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        reload_dirs="backend",
        reload=True,
    )
