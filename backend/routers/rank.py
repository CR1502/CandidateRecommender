from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from loguru import logger

from backend.dependencies import get_embedding_engine, get_summarizer
from backend.schemas.responses import RankResponse
from backend.services.pipeline import run_ranking_pipeline
from core.embeddings import EmbeddingEngine
from core.summarizer import CandidateSummarizer

router = APIRouter(tags=["ranking"])

ALLOWED_TYPES = {"application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain"}
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@router.post("/rank", response_model=RankResponse)
async def rank_candidates(
    job_description: str = Form(..., min_length=50, description="Full job description text"),
    files: List[UploadFile] = File(..., description="Resume files (PDF, DOCX, TXT)"),
    top_k: int = Form(default=10, ge=1, le=50),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
    summarizer: CandidateSummarizer = Depends(get_summarizer),
) -> RankResponse:
    """
    Rank uploaded resumes against a job description.

    Returns candidates sorted by composite score (semantic similarity +
    skill coverage + experience alignment).
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one resume file is required.",
        )

    # Validate file extensions before doing any heavy work
    for f in files:
        suffix = "." + (f.filename or "").rsplit(".", 1)[-1].lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type: {f.filename}. Allowed: PDF, DOCX, TXT.",
            )

    logger.info(f"Ranking request: {len(files)} files, top_k={top_k}")

    try:
        result = await run_ranking_pipeline(
            job_description=job_description,
            files=files,
            embedding_engine=embedding_engine,
            summarizer=summarizer,
            top_k=top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Ranking pipeline error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {e}",
        )

    logger.info(
        f"Ranking complete: {result.total_processed} candidates in {result.total_duration_ms}ms"
    )
    return result
