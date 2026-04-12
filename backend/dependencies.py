"""
Module-level singletons for the ML models.

Models are loaded once on first use (not at import time) so the FastAPI
process starts quickly and the heavy download only blocks the first request.
Use the get_* functions as FastAPI dependency injections.
"""

from __future__ import annotations

import sys
from pathlib import Path
from loguru import logger

# Make the src/ core modules importable from the backend package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import (
    EMBEDDING_MODEL_NAME,
    BGE_QUERY_PREFIX,
    SCORING_WEIGHTS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
)
from core.embeddings import EmbeddingEngine
from core.summarizer import CandidateSummarizer

_embedding_engine: EmbeddingEngine | None = None
_summarizer: CandidateSummarizer | None = None


def get_embedding_engine() -> EmbeddingEngine:
    global _embedding_engine
    if _embedding_engine is None:
        logger.info("Initialising embedding engine…")
        _embedding_engine = EmbeddingEngine(
            model_name=EMBEDDING_MODEL_NAME,
            query_prefix=BGE_QUERY_PREFIX,
            scoring_weights=SCORING_WEIGHTS,
        )
    return _embedding_engine


def get_summarizer() -> CandidateSummarizer:
    global _summarizer
    if _summarizer is None:
        logger.info("Initialising summarizer…")
        _summarizer = CandidateSummarizer(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            timeout=OLLAMA_TIMEOUT,
        )
    return _summarizer
