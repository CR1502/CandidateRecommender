from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


class ContactInfo(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None


class CandidateResult(BaseModel):
    rank: int
    candidate_name: str
    filename: str
    percentage_score: float
    composite_score: float
    similarity_score: float
    skill_coverage_score: float
    experience_score: float
    category: str
    category_emoji: str
    category_color: str
    matching_skills: List[str]
    fit_summary: str
    contact: ContactInfo


class RankResponse(BaseModel):
    total_processed: int
    total_duration_ms: int
    job_description: str
    candidates: List[CandidateResult]


class ExtractResponse(BaseModel):
    candidate_name: str
    skills: List[str]
    contact: ContactInfo


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    embedding_device: str
    ollama_available: bool
    ollama_model: str
    summary_mode: str  # "ollama" | "template"
