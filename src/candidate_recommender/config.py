"""
Configuration for the Candidate Recommendation Engine.

Every setting can be overridden with an environment variable of the same
name (case-insensitive) or a `.env` file in the working directory, e.g.

    EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
    OLLAMA_MODEL=mistral
    SCORING_WEIGHTS='{"semantic": 0.5, "skill_coverage": 0.4, "experience": 0.1}'
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/candidate_recommender/config.py → repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


class ScoringWeights(BaseModel):
    """Composite score weights — must sum to 1.0.

    semantic:       overall semantic match via embeddings
    skill_coverage: fraction of required skills found in the resume
    experience:     whether stated years of experience meets the job requirement
    """

    semantic: float = 0.60
    skill_coverage: float = 0.30
    experience: float = 0.10

    @model_validator(mode="after")
    def _sums_to_one(self) -> ScoringWeights:
        total = self.semantic + self.skill_coverage + self.experience
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"scoring weights must sum to 1.0, got {total:.3f}")
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Embedding model — bge-small (~130MB) is a big step up from MiniLM with a
    # small size increase. For higher accuracy use BAAI/bge-base-en-v1.5 (~420MB).
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # BGE models retrieve better when queries carry this prefix (passages don't).
    bge_query_prefix: str = "Represent this sentence for searching relevant passages: "

    # Ollama — free local LLM inference. Install from https://ollama.com, then
    # `ollama pull gemma4:12b`. Any Ollama model name works, including GGUF
    # builds from Hugging Face ("hf.co/<org>/<repo>:<quant>").
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:12b"
    ollama_timeout: int = 180  # seconds per request
    # Context window. Ollama's default (4096) can cut off a long resume; each
    # extra 2K tokens costs memory (~0.5GB on a 12B model), so don't overshoot.
    llm_num_ctx: int = 6144
    # Candidates assessed at once. On one GPU, 2 was only ~16% faster than 1
    # and doubles the context memory; raise it on machines with memory to spare.
    llm_concurrency: int = 1
    # Hide names, contact details, links, and location from the LLM.
    redact_pii: bool = True

    scoring_weights: ScoringWeights = ScoringWeights()
    # Raw cosine similarity is rescaled from [floor, ceiling] onto 0–1. Model
    # specific: measured on eval/ for bge-small (unrelated pairs ~0.45–0.6,
    # strong matches ~0.8). Re-tune with eval/run_eval.py if you change models.
    semantic_floor: float = 0.45
    semantic_ceiling: float = 0.85
    # Resume chunking for the embedding model. Without it the model truncates
    # each resume at ~512 tokens (about one page). Chunk scores are combined
    # with max, mean, or max_mean (half best chunk, half average). 0 disables.
    # Chosen on eval/: see eval/README.md.
    chunk_words: int = 250
    chunk_overlap_words: int = 50
    chunk_aggregation: Literal["max", "mean", "max_mean"] = "max_mean"

    # Uploads
    max_file_size_mb: int = 10
    max_files_per_upload: int = 20
    top_candidates_count: int = 10

    # Server
    log_level: str = "INFO"
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]
    # Built React app, served by FastAPI when present (`npm run build`).
    frontend_dist: Path = REPO_ROOT / "frontend" / "dist"


@lru_cache
def get_settings() -> Settings:
    return Settings()


class Category(NamedTuple):
    min_pct: float
    label: str
    emoji: str
    color: str


# Candidate categories, best first. Thresholds are on the composite 0–100 scale.
CANDIDATE_CATEGORIES: tuple[Category, ...] = (
    Category(85, "Perfect Match", "🌟", "#00D26A"),
    Category(70, "Ideal Candidate", "⭐", "#4CAF50"),
    Category(50, "Good Candidate", "✅", "#FFA726"),
    Category(25, "Okay Candidate", "👍", "#FF9800"),
    Category(0, "Not Recommended", "❌", "#F44336"),
)

ALLOWED_EXTENSIONS = ("pdf", "docx", "txt")
