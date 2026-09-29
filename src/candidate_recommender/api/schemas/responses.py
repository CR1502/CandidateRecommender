from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ContactInfo(BaseModel):
    email: str | None = None
    phone: str | None = None
    linkedin: str | None = None
    github: str | None = None
    location: str | None = None
    website: str | None = None


class CandidateResult(BaseModel):
    rank: int
    candidate_name: str
    filename: str
    percentage_score: float
    composite_score: float
    similarity_score: float  # raw cosine similarity
    semantic_score: float  # calibrated 0–1 semantic match (what the composite uses)
    skill_coverage_score: float | None  # None: the job lists no recognisable skills
    experience_score: float | None  # None: the job states no years of experience
    category: str
    category_emoji: str
    category_color: str
    matching_skills: list[str]
    fit_summary: str
    strengths: list[str] = []
    gaps: list[str] = []
    recommendation: Literal["Strong Yes", "Yes", "Maybe", "No"] | None = None
    summary_source: Literal["llm", "template"] = "template"
    contact: ContactInfo


class RankResponse(BaseModel):
    total_processed: int
    total_duration_ms: int
    job_description: str
    candidates: list[CandidateResult]


class ExtractResponse(BaseModel):
    candidate_name: str
    skills: list[str]
    contact: ContactInfo


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    embedding_device: str
    ollama_available: bool
    ollama_model: str
    summary_mode: str  # "ollama" | "template"
