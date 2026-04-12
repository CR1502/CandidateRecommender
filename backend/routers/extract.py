from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from loguru import logger

from backend.schemas.responses import ExtractResponse
from backend.services.pipeline import run_extract_pipeline

router = APIRouter(tags=["extract"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@router.post("/extract", response_model=ExtractResponse)
async def extract_resume(
    file: UploadFile = File(..., description="Single resume file (PDF, DOCX, TXT)"),
) -> ExtractResponse:
    """
    Extract contact information and skills from a single resume without ranking.
    Useful for quick candidate scans or pre-processing.
    """
    suffix = "." + (file.filename or "").rsplit(".", 1)[-1].lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.filename}. Allowed: PDF, DOCX, TXT.",
        )

    try:
        return await run_extract_pipeline(file)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Extract pipeline error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {e}",
        )
