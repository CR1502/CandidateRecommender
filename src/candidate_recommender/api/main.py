"""
FastAPI entry point for the Candidate Recommender backend.

Run (development):
    uv run uvicorn candidate_recommender.api.main:app --reload --port 8000

The frontend (React/Vite) runs separately on http://localhost:5173 in dev.
In production, `npm run build` outputs to frontend/dist/ which this app
serves as static files — no separate process needed.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Scope

from candidate_recommender.api.routers import extract, health, rank
from candidate_recommender.config import get_settings
from candidate_recommender.core.embeddings import EmbeddingEngine
from candidate_recommender.core.summarizer import CandidateSummarizer

settings = get_settings()

logger.remove()
logger.add(sys.stderr, level=settings.log_level.upper())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the models once at startup so no request pays the cold-start cost."""
    logger.info("Loading models…")
    app.state.embedding_engine = await asyncio.to_thread(
        EmbeddingEngine,
        model_name=settings.embedding_model,
        query_prefix=settings.bge_query_prefix,
        scoring_weights=settings.scoring_weights.model_dump(),
    )
    app.state.summarizer = await asyncio.to_thread(
        CandidateSummarizer,
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout,
    )
    logger.info("Candidate Recommender API ready — docs at /api/docs")
    yield


app = FastAPI(
    title="Candidate Recommender API",
    description="Rank and assess job candidates using local AI models.",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# CORS — allow the Vite dev server in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(rank.router, prefix="/api")
app.include_router(extract.router, prefix="/api")
app.include_router(health.router, prefix="/api")


class SPAStaticFiles(StaticFiles):
    """
    Static files with a single-page-app fallback: unknown page routes get
    index.html, so client-side routes like /results survive a page refresh.
    API paths and missing files (anything with an extension) still 404.
    """

    async def get_response(self, path: str, scope: Scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as e:
            is_api = path == "api" or path.startswith("api/")
            looks_like_file = "." in path.rsplit("/", 1)[-1]
            if e.status_code != 404 or is_api or looks_like_file:
                raise
            return await super().get_response("index.html", scope)


# Serve built frontend in production (after `npm run build`)
if settings.frontend_dist.is_dir():
    app.mount("/", SPAStaticFiles(directory=settings.frontend_dist, html=True), name="static")
    logger.info(f"Serving frontend from {settings.frontend_dist}")


if __name__ == "__main__":
    uvicorn.run("candidate_recommender.api.main:app", host="0.0.0.0", port=8000, reload=True)
