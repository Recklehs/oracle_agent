from fastapi import FastAPI


app = FastAPI(title="Oracle Agent")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
