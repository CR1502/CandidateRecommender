"""
FastAPI entry point for the Candidate Recommender backend.

Run (development):
    uvicorn backend.main:app --reload --port 8000

The frontend (React/Vite) runs separately on http://localhost:5173 in dev.
In production, `npm run build` outputs to frontend/dist/ which this app
serves as static files — no separate process needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backend.routers import extract, health, rank

app = FastAPI(
    title="Candidate Recommender API",
    description="Rank and assess job candidates using local AI models.",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# CORS — allow Vite dev server in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(rank.router, prefix="/api")
app.include_router(extract.router, prefix="/api")
app.include_router(health.router, prefix="/api")


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Candidate Recommender API starting up")
    logger.info("API docs available at http://localhost:8000/api/docs")


# Serve built frontend in production (after `npm run build`)
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="static")
    logger.info(f"Serving frontend from {_frontend_dist}")


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
