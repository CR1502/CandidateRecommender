from __future__ import annotations

from fastapi import APIRouter, Depends

from candidate_recommender.api.dependencies import get_embedding_engine, get_summarizer
from candidate_recommender.api.schemas.responses import HealthResponse
from candidate_recommender.core.embeddings import EmbeddingEngine
from candidate_recommender.core.summarizer import CandidateSummarizer

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check(
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
    summarizer: CandidateSummarizer = Depends(get_summarizer),
) -> HealthResponse:
    """Return current model status and Ollama availability."""
    info = embedding_engine.get_model_info()
    return HealthResponse(
        status="ok",
        embedding_model=info.get("model_name", "unknown"),
        embedding_device=info.get("device", "unknown"),
        ollama_available=summarizer._ollama_available,
        ollama_model=summarizer.model,
        summary_mode="ollama" if summarizer._ollama_available else "template",
    )
