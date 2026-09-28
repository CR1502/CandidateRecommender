"""
Embedding generation, composite scoring, and candidate ranking.
"""

import hashlib
from typing import Any

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from candidate_recommender.config import CANDIDATE_CATEGORIES, Settings

from .experience import candidate_years, stated_years
from .text_cleaner import TextCleaner

_text_cleaner = TextCleaner()

_UNKNOWN_EXPERIENCE_SCORE = 0.3

# How per-chunk similarities combine into one resume score.
_AGGREGATIONS = {
    "max": lambda s: float(np.max(s)),
    "mean": lambda s: float(np.mean(s)),
    "max_mean": lambda s: float(0.5 * np.max(s) + 0.5 * np.mean(s)),
}


def split_into_chunks(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    """
    Split text into overlapping word windows. The embedding model only reads
    ~512 tokens (~380 words), so without chunking anything past the first page
    of a resume — often the skills section — is invisible to it.
    """
    words = text.split()
    if chunk_words <= 0 or len(words) <= chunk_words:
        return [text]
    step = max(chunk_words - overlap_words, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_words]))
        if start + chunk_words >= len(words):
            break
    return chunks


class EmbeddingEngine:
    """
    Generate embeddings and rank candidates using a composite score:

        relevance = (0.60 * semantic + 0.30 * skill_coverage) / 0.90
        score     =  0.60 * semantic
                   + 0.30 * skill_coverage
                   + 0.10 * experience * relevance

    - `semantic` is cosine similarity rescaled from the model's working range
      [semantic_floor, semantic_ceiling] onto 0–1; raw BGE cosine between even
      unrelated documents is ~0.45, which would otherwise hand everyone points.
    - A component with no signal (the job lists no known skills, or states no
      years of experience) is left out and the remaining weights renormalised,
      rather than scored as a "neutral" 0.5 that inflates every candidate.
    - Experience is scaled by relevance, so ten years in an unrelated field
      can't lift a candidate — only relevant experience counts.

    This gives a much more realistic ranking than raw cosine similarity,
    because a Java developer whose resume talks about "software development"
    will score semantically similar to a Python job but low on skill coverage.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
        scoring_weights: dict[str, float] | None = None,
        semantic_floor: float = 0.45,
        semantic_ceiling: float = 0.85,
        chunk_words: int = 250,
        chunk_overlap_words: int = 50,
        chunk_aggregation: str = "max_mean",
    ):
        if semantic_ceiling <= semantic_floor:
            raise ValueError("semantic_ceiling must be greater than semantic_floor")
        self.model_name = model_name
        self.semantic_floor = semantic_floor
        self.semantic_ceiling = semantic_ceiling
        if chunk_aggregation not in _AGGREGATIONS:
            raise ValueError(f"chunk_aggregation must be one of {sorted(_AGGREGATIONS)}")
        self.chunk_words = chunk_words
        self.chunk_overlap_words = chunk_overlap_words
        self.chunk_aggregation = chunk_aggregation
        self.query_prefix = query_prefix
        self.scoring_weights = scoring_weights or {
            "semantic": 0.60,
            "skill_coverage": 0.30,
            "experience": 0.10,
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: SentenceTransformer | None = None
        self._load_model()

    @classmethod
    def from_settings(cls, settings: Settings) -> "EmbeddingEngine":
        """Build the engine the way the API does (also used by eval/run_eval.py)."""
        return cls(
            model_name=settings.embedding_model,
            query_prefix=settings.bge_query_prefix,
            scoring_weights=settings.scoring_weights.model_dump(),
            semantic_floor=settings.semantic_floor,
            semantic_ceiling=settings.semantic_ceiling,
            chunk_words=settings.chunk_words,
            chunk_overlap_words=settings.chunk_overlap_words,
            chunk_aggregation=settings.chunk_aggregation,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
            logger.info(f"Embedding model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Could not load embedding model: {e}") from e

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    def generate_embedding(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        Generate a single embedding.

        For BGE models, queries (job descriptions) benefit from a prefix that
        tells the model this is a retrieval query, not a passage to index.
        """
        if not text or not text.strip():
            raise ValueError("Input text is empty")

        input_text = f"{self.query_prefix}{text}" if is_query else text

        return self.model.encode(
            input_text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2-normalise so dot product == cosine
        )

    def generate_embeddings_batch(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Generate embeddings for a list of texts in one batched call."""
        valid = [t for t in texts if t and t.strip()]
        if not valid:
            raise ValueError("All provided texts are empty")

        if is_query:
            valid = [f"{self.query_prefix}{t}" for t in valid]

        logger.info(f"Encoding {len(valid)} texts (batch)")
        return self.model.encode(
            valid,
            convert_to_numpy=True,
            show_progress_bar=len(valid) > 5,
            batch_size=32,
            normalize_embeddings=True,
        )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _calibrate_semantic(self, cosine: float) -> float:
        """Map raw cosine from the model's working range onto 0–1."""
        span = self.semantic_ceiling - self.semantic_floor
        return float(min(max((cosine - self.semantic_floor) / span, 0.0), 1.0))

    def _skill_coverage_score(self, job_text: str, resume_text: str) -> float | None:
        """
        Weighted share of the job's skills that the resume demonstrates
        (0.0–1.0), or None when the job lists no recognisable skills.
        Nice-to-have skills weigh less; skills only listed (not described in
        work history) earn partial credit. See TextCleaner.skill_evidence.
        """
        weights = _text_cleaner.job_skill_weights(job_text)
        if not weights:
            return None

        evidence = _text_cleaner.skill_evidence(resume_text)
        earned = sum(w * evidence.get(skill, 0.0) for skill, w in weights.items())
        return earned / sum(weights.values())

    def _experience_signal(self, job_text: str, resume_text: str) -> float | None:
        """
        Heuristic: does the candidate's stated years of experience meet or
        exceed what the job asks for? Returns 0.0–1.0, or None when the job
        states no requirement.
        """
        required = stated_years(job_text, use_lower_bound=True)
        if not required:
            return None

        candidate = candidate_years(resume_text)
        if not candidate:
            return _UNKNOWN_EXPERIENCE_SCORE

        # Proportional credit, floored so that stating a few years never
        # scores below stating nothing at all (keeps the score monotonic).
        return max(_UNKNOWN_EXPERIENCE_SCORE, min(candidate / required, 1.0))

    def _composite_score(
        self,
        semantic: float,
        skill: float | None,
        exp: float | None,
    ) -> float:
        """
        Combine calibrated semantic, skill coverage, and experience into one
        score. Missing components (None) are dropped and weights renormalised.
        """
        w = self.scoring_weights
        rel_parts = [(w["semantic"], semantic)]
        if skill is not None:
            rel_parts.append((w["skill_coverage"], skill))
        rel_weight = sum(weight for weight, _ in rel_parts)
        relevance = sum(weight * value for weight, value in rel_parts) / rel_weight

        if exp is None:
            score = relevance
        else:
            total = rel_weight + w["experience"]
            score = (rel_weight * relevance + w["experience"] * exp * relevance) / total
        return float(min(max(score, 0.0), 1.0))

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.md5(text.strip().encode()).hexdigest()

    def _deduplicate(self, resumes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate resumes (same content hash). Keeps first occurrence."""
        seen: set = set()
        unique = []
        for r in resumes:
            h = self._text_hash(r.get("text", ""))
            if h in seen:
                logger.warning(f"Duplicate resume detected and removed: {r.get('filename', '?')}")
            else:
                seen.add(h)
                unique.append(r)
        return unique

    # ------------------------------------------------------------------
    # Main ranking method
    # ------------------------------------------------------------------

    def rank_candidates(
        self,
        job_description: str,
        resumes: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Score and rank candidates against a job description.

        Each candidate dict must have at minimum:
            {'text': str, 'candidate_name': str, 'filename': str}

        Returns a sorted list (best first) with these extra fields added:
            similarity_score, skill_coverage_score, experience_score,
            composite_score, percentage_score, category, category_emoji,
            category_color, rank
        """
        if not job_description:
            raise ValueError("Job description is empty")
        if not resumes:
            raise ValueError("No resumes provided")

        resumes = self._deduplicate(resumes)

        # Encode job description with query prefix
        logger.info("Encoding job description")
        job_emb = self.generate_embedding(job_description, is_query=True).reshape(1, -1)

        # Encode all resume chunks (passages — no prefix) in one batch
        chunks_per_resume = [
            split_into_chunks(r["text"], self.chunk_words, self.chunk_overlap_words)
            for r in resumes
        ]
        flat_chunks = [chunk for chunks in chunks_per_resume for chunk in chunks]
        logger.info(f"Encoding {len(resumes)} resumes ({len(flat_chunks)} chunks)")
        chunk_embs = self.generate_embeddings_batch(flat_chunks, is_query=False)

        # Cosine similarities (embeddings are L2-normalised so this is just dot
        # product), then combine each resume's chunk scores into one.
        chunk_scores = cosine_similarity(job_emb, chunk_embs).flatten()
        aggregate = _AGGREGATIONS[self.chunk_aggregation]
        semantic_scores, offset = [], 0
        for chunks in chunks_per_resume:
            semantic_scores.append(aggregate(chunk_scores[offset : offset + len(chunks)]))
            offset += len(chunks)

        results = []
        for resume, sem_score in zip(resumes, semantic_scores, strict=True):
            semantic = self._calibrate_semantic(float(sem_score))
            # Raw text keeps line breaks, which skill evidence and date parsing use.
            source = resume.get("raw_text") or resume["text"]
            skill_cov = self._skill_coverage_score(job_description, source)
            exp_sig = self._experience_signal(job_description, source)
            composite = self._composite_score(semantic, skill_cov, exp_sig)
            pct = composite * 100

            category, emoji, color = self._classify(pct)

            results.append(
                {
                    **resume,
                    "similarity_score": float(sem_score),
                    "semantic_score": round(semantic, 3),
                    "skill_coverage_score": None if skill_cov is None else round(skill_cov, 3),
                    "experience_score": None if exp_sig is None else round(exp_sig, 3),
                    "composite_score": round(composite, 4),
                    "percentage_score": round(pct, 1),
                    "category": category,
                    "category_emoji": emoji,
                    "category_color": color,
                    "rank": 0,  # set after sort
                }
            )

        # Stable sort: composite desc, then original index as tiebreaker
        for i, r in enumerate(results):
            r["_original_index"] = i
        results.sort(key=lambda x: (-x["composite_score"], x["_original_index"]))
        for r in results:
            del r["_original_index"]

        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results[:top_k]

    @staticmethod
    def _classify(pct: float):
        """Return (category_label, emoji, hex_color) for a percentage score."""
        for category in CANDIDATE_CATEGORIES:
            if pct >= category.min_pct:
                return category.label, category.emoji, category.color
        lowest = CANDIDATE_CATEGORIES[-1]
        return lowest.label, lowest.emoji, lowest.color

    # ------------------------------------------------------------------
    # Skill matching (public helper for UI layer)
    # ------------------------------------------------------------------

    def find_matching_skills(self, job_text: str, resume_text: str) -> list[str]:
        """Return skills that appear in both the job description and the resume."""
        job_skills = set(_text_cleaner.extract_key_skills(job_text))
        resume_skills = set(_text_cleaner.extract_key_skills(resume_text))
        return sorted(job_skills & resume_skills)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict[str, Any]:
        if not self.model:
            return {"error": "Model not loaded"}
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_seq_length": self.model.max_seq_length,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "scoring_weights": self.scoring_weights,
        }
