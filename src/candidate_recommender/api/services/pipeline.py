"""
Main processing pipeline: files → text → embeddings → rank → enrich → assessments.

`run_ranking_pipeline` returns the result; `stream_ranking_pipeline` yields
progress events while it works (used by POST /api/rank/stream), since LLM
assessments can take several seconds per candidate.
"""

from __future__ import annotations

import asyncio
import io
import time
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import UploadFile
from loguru import logger

from candidate_recommender.api.schemas.responses import (
    CandidateResult,
    ContactInfo,
    ExtractResponse,
    RankResponse,
)
from candidate_recommender.config import get_settings
from candidate_recommender.core.embeddings import EmbeddingEngine
from candidate_recommender.core.enricher import enrich_candidate
from candidate_recommender.core.file_processor import FileProcessor
from candidate_recommender.core.summarizer import CandidateSummarizer
from candidate_recommender.core.text_cleaner import TextCleaner

_settings = get_settings()
_file_processor = FileProcessor(max_file_size_mb=_settings.max_file_size_mb)
_text_cleaner = TextCleaner()

# progress(stage, done, total). Stages, in order: extracting, ranking,
# enriching, assessing.
Progress = Callable[[str, int, int], None]


def _no_progress(stage: str, done: int, total: int) -> None:
    pass


def _run_ranking_sync(
    job_description: str,
    file_payloads: list[tuple[str, bytes]],
    embedding_engine: EmbeddingEngine,
    summarizer: CandidateSummarizer,
    top_k: int,
    progress: Progress = _no_progress,
) -> RankResponse:
    start = time.time()

    # --- 1. Extract text ---
    resumes: list[dict] = []
    for i, (filename, content) in enumerate(file_payloads):
        progress("extracting", i, len(file_payloads))
        file_obj = io.BytesIO(content)
        try:
            is_valid, err = _file_processor.validate_file(file_obj, filename)
            if not is_valid:
                logger.warning(f"Skipping {filename}: {err}")
                continue
            file_obj.seek(0)
            raw_text, candidate_name = _file_processor.process_file(file_obj, filename)
            if not raw_text or len(raw_text.strip()) < 50:
                logger.warning(f"Skipping {filename}: extracted text too short")
                continue
            resumes.append(
                {
                    "filename": filename,
                    "candidate_name": candidate_name,
                    "raw_text": raw_text,
                    "text": _text_cleaner.prepare_for_embedding(raw_text),
                }
            )
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
    progress("extracting", len(file_payloads), len(file_payloads))

    if not resumes:
        raise ValueError(
            "No valid resume text could be extracted. "
            "Check that the files are readable PDFs, DOCX, or plain text."
        )

    # --- 2. Rank ---
    progress("ranking", 0, 1)
    clean_jd = _text_cleaner.prepare_for_embedding(job_description)
    ranked = embedding_engine.rank_candidates(clean_jd, resumes, top_k=top_k)
    progress("ranking", 1, 1)

    # --- 3. Contact info and keyword skill matches (on raw text) ---
    jd_skills = {s.lower() for s in _text_cleaner.extract_key_skills(clean_jd)}
    for candidate in ranked:
        raw = candidate.get("raw_text", candidate["text"])
        candidate["contact"] = _text_cleaner.extract_contact_details(raw)
        candidate["matching_skills"] = [
            s for s in _text_cleaner.extract_key_skills(raw) if s.lower() in jd_skills
        ]

    # --- 4. URL enrichment: follow GitHub + portfolio links, candidates in parallel ---
    # Stored on the candidate itself: names (and filenames) can collide.
    def enrich(candidate: dict[str, Any]) -> None:
        raw = candidate.get("raw_text", candidate["text"])
        try:
            ctx = enrich_candidate(raw, candidate.get("contact", {}))
        except Exception as e:
            logger.warning(f"Enrichment failed for {candidate['filename']}: {e}")
            ctx = ""
        if ctx:
            candidate["enriched_context"] = ctx
            logger.info(
                f"Enriched {candidate['candidate_name']} ({len(ctx)} chars from online profiles)"
            )

    progress("enriching", 0, len(ranked))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(enrich, ranked), start=1):
            progress("enriching", i, len(ranked))

    # --- 5. Assessments (LLM or template), with enriched context ---
    progress("assessing", 0, len(ranked))
    ranked = summarizer.batch_assess(
        ranked, clean_jd, progress=lambda done, total: progress("assessing", done, total)
    )

    duration_ms = int((time.time() - start) * 1000)
    return RankResponse(
        total_processed=len(resumes),
        total_duration_ms=duration_ms,
        job_description=job_description,
        candidates=[_to_candidate_result(c) for c in ranked],
    )


async def _read_uploads(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    return [(f.filename or "upload", await f.read()) for f in files]


async def run_ranking_pipeline(
    job_description: str,
    files: list[UploadFile],
    embedding_engine: EmbeddingEngine,
    summarizer: CandidateSummarizer,
    top_k: int = 10,
) -> RankResponse:
    file_payloads = await _read_uploads(files)
    return await asyncio.to_thread(
        _run_ranking_sync, job_description, file_payloads, embedding_engine, summarizer, top_k
    )


async def stream_ranking_pipeline(
    job_description: str,
    files: list[UploadFile],
    embedding_engine: EmbeddingEngine,
    summarizer: CandidateSummarizer,
    top_k: int = 10,
) -> AsyncIterator[dict[str, Any]]:
    """
    Run the pipeline in a worker thread, yielding
    {"event": "progress", "stage", "done", "total"} as it goes, then a final
    {"event": "result", "data": RankResponse} — or raising what the
    pipeline raised.
    """
    file_payloads = await _read_uploads(files)
    loop = asyncio.get_running_loop()
    events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def progress(stage: str, done: int, total: int) -> None:
        event = {"event": "progress", "stage": stage, "done": done, "total": total}
        loop.call_soon_threadsafe(events.put_nowait, event)

    task = asyncio.create_task(
        asyncio.to_thread(
            _run_ranking_sync,
            job_description,
            file_payloads,
            embedding_engine,
            summarizer,
            top_k,
            progress,
        )
    )
    while not task.done():
        getter = asyncio.create_task(events.get())
        finished, _ = await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
        if getter in finished:
            yield getter.result()
        else:
            getter.cancel()
    while not events.empty():
        yield events.get_nowait()
    yield {"event": "result", "data": task.result()}  # re-raises pipeline errors


def _run_extract_sync(
    filename: str, content: bytes, summarizer: CandidateSummarizer
) -> ExtractResponse:
    file_obj = io.BytesIO(content)
    is_valid, err = _file_processor.validate_file(file_obj, filename)
    if not is_valid:
        raise ValueError(err)
    file_obj.seek(0)
    raw_text, candidate_name = _file_processor.process_file(file_obj, filename)
    return ExtractResponse(
        candidate_name=candidate_name,
        skills=summarizer.extract_skills(raw_text),
        contact=ContactInfo(**_text_cleaner.extract_contact_details(raw_text)),
    )


async def run_extract_pipeline(
    file: UploadFile, summarizer: CandidateSummarizer
) -> ExtractResponse:
    content = await file.read()
    return await asyncio.to_thread(
        _run_extract_sync, file.filename or "upload", content, summarizer
    )


def _to_candidate_result(c: dict) -> CandidateResult:
    raw_contact = c.get("contact", {})
    contact = ContactInfo(**raw_contact) if isinstance(raw_contact, dict) else ContactInfo()
    return CandidateResult(
        rank=c["rank"],
        candidate_name=c["candidate_name"],
        filename=c.get("filename", ""),
        percentage_score=c["percentage_score"],
        composite_score=c["composite_score"],
        similarity_score=c["similarity_score"],
        semantic_score=c["semantic_score"],
        skill_coverage_score=c["skill_coverage_score"],
        experience_score=c["experience_score"],
        category=c["category"],
        category_emoji=c["category_emoji"],
        category_color=c["category_color"],
        matching_skills=c.get("matching_skills", []),
        fit_summary=c.get("fit_summary", ""),
        strengths=c.get("strengths", []),
        gaps=c.get("gaps", []),
        recommendation=c.get("recommendation"),
        summary_source=c.get("summary_source", "template"),
        contact=contact,
    )
