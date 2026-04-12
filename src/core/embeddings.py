"""
Embedding generation, composite scoring, and candidate ranking.
"""

import hashlib
import re
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
import torch


class EmbeddingEngine:
    """
    Generate embeddings and rank candidates using a composite score:

        score = 0.60 * semantic_similarity
              + 0.30 * skill_coverage
              + 0.10 * experience_signal

    This gives a much more realistic ranking than raw cosine similarity,
    because a Java developer whose resume talks about "software development"
    will score semantically similar to a Python job but low on skill coverage.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
        scoring_weights: Optional[Dict[str, float]] = None,
    ):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.scoring_weights = scoring_weights or {
            "semantic": 0.60,
            "skill_coverage": 0.30,
            "experience": 0.10,
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[SentenceTransformer] = None
        self._load_model()

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

    def generate_embeddings_batch(
        self, texts: List[str], is_query: bool = False
    ) -> np.ndarray:
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

    def _skill_coverage_score(
        self, job_text: str, resume_text: str
    ) -> float:
        """
        Fraction of required job skills that appear in the resume.
        Returns 0.0–1.0.
        """
        from .text_cleaner import TextCleaner

        cleaner = TextCleaner()
        required = set(cleaner.extract_required_skills(job_text))
        if not required:
            return 0.5  # No extractable required skills → neutral

        present = set(cleaner.extract_key_skills(resume_text))
        overlap = required & present
        return len(overlap) / len(required)

    def _experience_signal(self, job_text: str, resume_text: str) -> float:
        """
        Heuristic: does the candidate's stated years of experience meet or
        exceed what the job asks for?  Returns 0.0–1.0.
        """
        years_pattern = re.compile(r'(\d+)\+?\s*years?', re.IGNORECASE)

        job_years = [int(m) for m in years_pattern.findall(job_text)]
        resume_years = [int(m) for m in years_pattern.findall(resume_text)]

        if not job_years:
            return 0.5  # Job doesn't state a requirement → neutral

        required = max(job_years)
        candidate = max(resume_years) if resume_years else 0

        if candidate >= required:
            return 1.0
        elif candidate == 0:
            return 0.3
        else:
            # Proportional credit
            return min(candidate / required, 1.0)

    def _composite_score(
        self,
        semantic: float,
        job_text: str,
        resume_text: str,
    ) -> float:
        """Combine semantic, skill coverage, and experience into one score."""
        w = self.scoring_weights
        skill = self._skill_coverage_score(job_text, resume_text)
        exp = self._experience_signal(job_text, resume_text)

        score = (
            w["semantic"] * semantic
            + w["skill_coverage"] * skill
            + w["experience"] * exp
        )
        return float(min(max(score, 0.0), 1.0))

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.md5(text.strip().encode()).hexdigest()

    def _deduplicate(
        self, resumes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate resumes (same content hash). Keeps first occurrence."""
        seen: set = set()
        unique = []
        for r in resumes:
            h = self._text_hash(r.get("text", ""))
            if h in seen:
                logger.warning(
                    f"Duplicate resume detected and removed: {r.get('filename', '?')}"
                )
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
        resumes: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
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

        # Encode all resumes (passages — no prefix)
        resume_texts = [r["text"] for r in resumes]
        logger.info(f"Encoding {len(resume_texts)} resumes")
        resume_embs = self.generate_embeddings_batch(resume_texts, is_query=False)

        # Cosine similarities (embeddings are L2-normalised so this is just dot product)
        semantic_scores = cosine_similarity(job_emb, resume_embs).flatten()

        results = []
        for i, (resume, sem_score) in enumerate(zip(resumes, semantic_scores)):
            composite = self._composite_score(
                float(sem_score), job_description, resume["text"]
            )
            skill_cov = self._skill_coverage_score(job_description, resume["text"])
            exp_sig = self._experience_signal(job_description, resume["text"])
            pct = composite * 100

            category, emoji, color = self._classify(pct)

            results.append({
                **resume,
                "similarity_score": float(sem_score),
                "skill_coverage_score": round(skill_cov, 3),
                "experience_score": round(exp_sig, 3),
                "composite_score": round(composite, 4),
                "percentage_score": round(pct, 1),
                "category": category,
                "category_emoji": emoji,
                "category_color": color,
                "rank": 0,  # set after sort
            })

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
        if pct >= 85:
            return "Perfect Match",    "🌟", "#00D26A"
        elif pct >= 70:
            return "Ideal Candidate",  "⭐", "#4CAF50"
        elif pct >= 50:
            return "Good Candidate",   "✅", "#FFA726"
        elif pct >= 25:
            return "Okay Candidate",   "👍", "#FF9800"
        else:
            return "Not Recommended",  "❌", "#F44336"

    # ------------------------------------------------------------------
    # Skill matching (public helper for UI layer)
    # ------------------------------------------------------------------

    def find_matching_skills(self, job_text: str, resume_text: str) -> List[str]:
        """Return skills that appear in both the job description and the resume."""
        from .text_cleaner import TextCleaner

        cleaner = TextCleaner()
        job_skills = set(cleaner.extract_key_skills(job_text))
        resume_skills = set(cleaner.extract_key_skills(resume_text))
        return sorted(job_skills & resume_skills)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_model_info(self) -> Dict[str, Any]:
        if not self.model:
            return {"error": "Model not loaded"}
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_seq_length": self.model.max_seq_length,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "scoring_weights": self.scoring_weights,
        }
