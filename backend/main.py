import uvicorn
from fastapi import FastAPI, Request

from .analyze import analyze_plan
from .pg import explain
from .predict import predict
from .recommend import recommend

app = FastAPI(debug=True)


@app.post("/explain")
async def post_explain(request: Request, analyze: bool = False):
    body = await request.body()
    sql = body.decode()

    plan = await explain(sql, analyze, format="JSON")
    prediction = predict(plan)
    analysis = analyze_plan(plan)
    recommendation = await recommend(sql)

    return {
        "plan": plan,
        "prediction": prediction,
        "actual": analysis,
        "recommendation": recommendation,
    }


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        reload_dirs="backend",
        reload=True,
    )
