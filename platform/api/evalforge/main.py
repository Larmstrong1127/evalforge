"""FastAPI app entry point. Run with: uvicorn evalforge.main:app --reload"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from evalforge.api import compare, prompts, ratings, runs, suites
from evalforge.db.engine import init_db
from evalforge.db.session import _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db(_engine)
    yield
    await _engine.dispose()


app = FastAPI(title="EvalForge API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(suites.router, prefix="/api/v1")
app.include_router(prompts.router, prefix="/api/v1")
app.include_router(runs.router, prefix="/api/v1")
app.include_router(ratings.router, prefix="/api/v1")
app.include_router(compare.router, prefix="/api/v1")
