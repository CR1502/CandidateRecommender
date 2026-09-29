from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from loguru import logger

from candidate_recommender.api.dependencies import get_embedding_engine, get_summarizer
from candidate_recommender.api.schemas.responses import RankResponse
from candidate_recommender.api.services.pipeline import (
    run_ranking_pipeline,
    stream_ranking_pipeline,
)
from candidate_recommender.config import ALLOWED_EXTENSIONS, get_settings
from candidate_recommender.core.embeddings import EmbeddingEngine
from candidate_recommender.core.summarizer import CandidateSummarizer

router = APIRouter(tags=["ranking"])


def _validate_upload(files: list[UploadFile], top_k: int | None) -> int:
    """Reject bad requests before any heavy work; return the effective top_k."""
    if not files:
        raise HTTPException(status_code=422, detail="At least one resume file is required.")

    settings = get_settings()
    if len(files) > settings.max_files_per_upload:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files: {len(files)}. Maximum is {settings.max_files_per_upload}.",
        )
    for f in files:
        suffix = (f.filename or "").rsplit(".", 1)[-1].lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type: {f.filename}. Allowed: PDF, DOCX, TXT.",
            )
    return top_k or settings.top_candidates_count


@router.post("/rank", response_model=RankResponse)
async def rank_candidates(
    job_description: str = Form(..., min_length=50, description="Full job description text"),
    files: list[UploadFile] = File(..., description="Resume files (PDF, DOCX, TXT)"),
    top_k: int | None = Form(default=None, ge=1, le=50),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
    summarizer: CandidateSummarizer = Depends(get_summarizer),
) -> RankResponse:
    """
    Rank uploaded resumes against a job description.

    Returns candidates sorted by composite score (semantic similarity +
    skill coverage + experience alignment). See /rank/stream for a version
    that reports progress while it works.
    """
    top_k = _validate_upload(files, top_k)
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
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception(f"Ranking pipeline error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Processing failed due to an internal error.",
        ) from e

    logger.info(
        f"Ranking complete: {result.total_processed} candidates in {result.total_duration_ms}ms"
    )
    return result


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post(
    "/rank/stream",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def rank_candidates_stream(
    job_description: str = Form(..., min_length=50, description="Full job description text"),
    files: list[UploadFile] = File(..., description="Resume files (PDF, DOCX, TXT)"),
    top_k: int | None = Form(default=None, ge=1, le=50),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
    summarizer: CandidateSummarizer = Depends(get_summarizer),
) -> StreamingResponse:
    """
    Same as /rank, streamed as Server-Sent Events:

    - `progress`: {"stage": "extracting" | "ranking" | "enriching" | "assessing", "done", "total"}
    - `result`: the RankResponse JSON (last event on success)
    - `error`: {"detail": "..."} (last event on failure)

    Request validation errors are returned as normal HTTP errors before the stream starts.
    """
    top_k = _validate_upload(files, top_k)
    logger.info(f"Streaming ranking request: {len(files)} files, top_k={top_k}")

    async def events() -> AsyncIterator[str]:
        try:
            async for event in stream_ranking_pipeline(
                job_description, files, embedding_engine, summarizer, top_k
            ):
                if event["event"] == "result":
                    yield _sse("result", event["data"].model_dump(mode="json"))
                else:
                    yield _sse("progress", {k: v for k, v in event.items() if k != "event"})
        except ValueError as e:
            yield _sse("error", {"detail": str(e)})
        except Exception as e:
            logger.exception(f"Ranking pipeline error: {e}")
            yield _sse("error", {"detail": "Processing failed due to an internal error."})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
