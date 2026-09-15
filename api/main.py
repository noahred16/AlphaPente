from fastapi import FastAPI

from api.routers import book

app = FastAPI(title="AlphaPente API")
app.include_router(book.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
