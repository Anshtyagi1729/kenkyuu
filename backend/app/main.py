import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import compare, library, papers, search
from app.storage import db, vector_store

# uvicorn configures its own loggers and leaves the root logger untouched, so without
# this every logger.info in app/ is silently dropped. That matters more here than in a
# typical service: this is an agent, and which tools it chose, how retrieval was
# graded, and whether a corrective retry fired are the only visible trace of its
# reasoning. Debugging a wrong answer without them is guesswork.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
# These two narrate every outbound HTTP call at INFO and drown out everything above.
for noisy in ("httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

app = FastAPI(title="Paper Agent")

# Local dev only: the Next.js frontend runs on a different port, so the
# browser treats it as a different origin. Tighten this before any real
# deployment - wide open is fine for a single-user local dev tool, not
# for anything reachable outside localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router)
app.include_router(compare.router)
app.include_router(papers.router)
app.include_router(library.router)


@app.on_event("startup")
def on_startup() -> None:
    db.init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": vector_store.count()}
