"""
Main processing pipeline: files → text → embeddings → rank → enrich → summaries.
"""

from __future__ import annotations

import asyncio
import io
import time

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
OLLAMA_BASE_URL = _settings.ollama_base_url
OLLAMA_MODEL = _settings.ollama_model

_file_processor = FileProcessor(max_file_size_mb=_settings.max_file_size_mb)
_text_cleaner = TextCleaner()


def _run_ranking_sync(
    job_description: str,
    file_payloads: list[tuple[str, bytes]],
    embedding_engine: EmbeddingEngine,
    summarizer: CandidateSummarizer,
    top_k: int,
) -> RankResponse:
    start = time.time()

    # --- 1. Extract text ---
    resumes: list[dict] = []
    for filename, content in file_payloads:
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

    if not resumes:
        raise ValueError(
            "No valid resume text could be extracted. "
            "Check that the files are readable PDFs, DOCX, or plain text."
        )

    # --- 2. Rank (uses dictionary skills for speed) ---
    clean_jd = _text_cleaner.prepare_for_embedding(job_description)
    ranked = embedding_engine.rank_candidates(clean_jd, resumes, top_k=top_k)

    # --- 3. Contact info (on raw text) ---
    for candidate in ranked:
        candidate["contact"] = _text_cleaner.extract_contact_details(
            candidate.get("raw_text", candidate["text"])
        )

    # --- 4. Skill extraction for display (LLM if available, dictionary otherwise) ---
    # JD skills extracted ONCE here, not once per candidate.
    # Both sides use the same extraction method so the intersection is meaningful.
    use_llm_skills = summarizer._ollama_available
    if use_llm_skills:
        jd_skills = set(
            _text_cleaner.extract_skills_with_llm(
                clean_jd, base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL
            )
        )
    else:
        jd_skills = set(_text_cleaner.extract_key_skills(clean_jd))

    for candidate in ranked:
        raw = candidate.get("raw_text", candidate["text"])
        if use_llm_skills:
            resume_skills = set(
                _text_cleaner.extract_skills_with_llm(
                    raw, base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL
                )
            )
        else:
            resume_skills = set(_text_cleaner.extract_key_skills(raw))
        # LLM names are canonicalised to registry names; compare case-insensitively too
        jd_lower = {s.lower() for s in jd_skills}
        candidate["matching_skills"] = sorted(s for s in resume_skills if s.lower() in jd_lower)

    # --- 5. URL enrichment: follow GitHub + portfolio links ---
    # Stored on the candidate itself: names (and filenames) can collide.
    for candidate in ranked:
        raw = candidate.get("raw_text", candidate["text"])
        contact = candidate.get("contact", {})
        try:
            ctx = enrich_candidate(raw, contact)
        except Exception as e:
            logger.warning(f"Enrichment failed for {candidate['filename']}: {e}")
            ctx = ""
        if ctx:
            candidate["enriched_context"] = ctx
            logger.info(
                f"Enriched {candidate['candidate_name']} ({len(ctx)} chars from online profiles)"
            )

    # --- 6. Summaries (with enriched context) ---
    ranked = summarizer.batch_generate_summaries(ranked, clean_jd)

    duration_ms = int((time.time() - start) * 1000)
    return RankResponse(
        total_processed=len(resumes),
        total_duration_ms=duration_ms,
        job_description=job_description,
        candidates=[_to_candidate_result(c) for c in ranked],
    )


async def run_ranking_pipeline(
    job_description: str,
    files: list[UploadFile],
    embedding_engine: EmbeddingEngine,
    summarizer: CandidateSummarizer,
    top_k: int = 10,
) -> RankResponse:
    file_payloads = []
    for f in files:
        content = await f.read()
        file_payloads.append((f.filename or "upload", content))

    return await asyncio.to_thread(
        _run_ranking_sync,
        job_description,
        file_payloads,
        embedding_engine,
        summarizer,
        top_k,
    )


def _run_extract_sync(filename: str, content: bytes) -> ExtractResponse:
    file_obj = io.BytesIO(content)
    is_valid, err = _file_processor.validate_file(file_obj, filename)
    if not is_valid:
        raise ValueError(err)
    file_obj.seek(0)
    raw_text, candidate_name = _file_processor.process_file(file_obj, filename)
    skills = _text_cleaner.extract_skills_with_llm(
        raw_text, base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL
    )
    contact = _text_cleaner.extract_contact_details(raw_text)
    return ExtractResponse(
        candidate_name=candidate_name,
        skills=skills,
        contact=ContactInfo(**contact),
    )


async def run_extract_pipeline(file: UploadFile) -> ExtractResponse:
    content = await file.read()
    return await asyncio.to_thread(_run_extract_sync, file.filename or "upload", content)


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
        contact=contact,
    )
