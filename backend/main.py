import uvicorn
from fastapi import FastAPI

app = FastAPI(debug=True)


@app.post("/")
async def post_explain():
    return {"Hello": "World"}


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        reload_dirs="backend",
        reload=True,
    )
