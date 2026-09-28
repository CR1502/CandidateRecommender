"""
FastAPI dependencies for the ML models.

The models are loaded once at startup by the lifespan handler in
`candidate_recommender.api.main` and kept on `app.state`; these functions
hand them to routes. Tests replace them via `app.dependency_overrides`.
"""

from __future__ import annotations

from fastapi import Request

from candidate_recommender.core.embeddings import EmbeddingEngine
from candidate_recommender.core.summarizer import CandidateSummarizer


def get_embedding_engine(request: Request) -> EmbeddingEngine:
    return request.app.state.embedding_engine


def get_summarizer(request: Request) -> CandidateSummarizer:
    return request.app.state.summarizer
