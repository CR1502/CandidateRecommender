from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from loguru import logger

from candidate_recommender.api.schemas.responses import ExtractResponse
from candidate_recommender.api.services.pipeline import run_extract_pipeline
from candidate_recommender.config import ALLOWED_EXTENSIONS

router = APIRouter(tags=["extract"])


@router.post("/extract", response_model=ExtractResponse)
async def extract_resume(
    file: UploadFile = File(..., description="Single resume file (PDF, DOCX, TXT)"),
) -> ExtractResponse:
    """
    Extract contact information and skills from a single resume without ranking.
    Useful for quick candidate scans or pre-processing.
    """
    suffix = (file.filename or "").rsplit(".", 1)[-1].lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.filename}. Allowed: PDF, DOCX, TXT.",
        )

    try:
        return await run_extract_pipeline(file)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception(f"Extract pipeline error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Extraction failed due to an internal error.",
        ) from e
