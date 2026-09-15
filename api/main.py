from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import book

app = FastAPI(title="AlphaPente API")

# The frontend (docs/) is served from a different origin (a local static file
# server, GitHub Pages, etc.), so the browser needs CORS to allow its fetch
# calls. Wide open since this only ever runs as a local dev tool.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(book.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
