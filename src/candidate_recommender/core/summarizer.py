"""
Candidate fit summarization via Ollama (free local LLM) with a deterministic
template fallback when Ollama is not running.

Setup:
    1. Install Ollama: https://ollama.com
    2. Pull a model: ollama pull llama3.2
    3. Start the server: ollama serve  (runs automatically on macOS after install)

Supported models (set OLLAMA_MODEL env var):
    llama3.2     — 3B, fast, good quality (default)
    mistral      — 7B, slower but noticeably better reasoning
    phi3         — 3.8B, very capable for its size
    gemma2       — 9B, strong analytical writing
"""

import re
from typing import Any

from loguru import logger

from .text_cleaner import TextCleaner

_text_cleaner = TextCleaner()

try:
    import requests as _requests

    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


class CandidateSummarizer:
    """
    Generate concise, factual fit assessments for candidates.

    Uses Ollama for LLM-backed summaries; falls back to a rule-based
    (but deterministic) template when Ollama is unavailable.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._ollama_available = self._check_ollama()

    # ------------------------------------------------------------------
    # Ollama connectivity
    # ------------------------------------------------------------------

    def _check_ollama(self) -> bool:
        if not _HAS_REQUESTS:
            logger.warning("requests library not installed; Ollama summaries disabled")
            return False
        try:
            resp = _requests.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
                if self.model not in models:
                    logger.warning(
                        f"Ollama is running but model '{self.model}' is not pulled. "
                        f"Run: ollama pull {self.model}"
                    )
                    return False
                logger.info(f"Ollama available with model '{self.model}'")
                return True
        except Exception as e:
            logger.info(f"Ollama not reachable ({e}); using template summaries")
        return False

    def _call_ollama(self, prompt: str, num_predict: int = 300) -> str:
        """POST to /api/generate and return the response text."""
        resp = _requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_predict": num_predict,
                },
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_fit_summary(
        self,
        job_description: str,
        resume_text: str,
        composite_score: float,
        matching_skills: list[str] | None = None,
        skill_coverage: float = 0.0,
        experience_score: float = 0.0,
        enriched_context: str = "",
    ) -> str:
        """
        Generate a detailed, evidence-based assessment of candidate fit.

        Args:
            job_description:   Full job description text.
            resume_text:       Full resume text.
            composite_score:   0–1 composite match score.
            matching_skills:   Skills found in both JD and resume.
            skill_coverage:    Fraction of required skills matched (0–1).
            experience_score:  Experience heuristic score (0–1).
            enriched_context:  Extra context from GitHub / portfolio URLs.

        Returns:
            Multi-sentence plain-text assessment.
        """
        if self._ollama_available:
            try:
                return self._generate_ollama_summary(
                    job_description,
                    resume_text,
                    composite_score,
                    matching_skills,
                    skill_coverage,
                    experience_score,
                    enriched_context,
                )
            except Exception as e:
                logger.warning(f"Ollama summary failed: {e}; falling back to template")

        return self._generate_template_summary(
            job_description,
            resume_text,
            composite_score,
            matching_skills,
            skill_coverage,
            experience_score,
            enriched_context,
        )

    def batch_generate_summaries(
        self,
        candidates: list[dict[str, Any]],
        job_description: str,
        enriched_contexts: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Add a 'fit_summary' field to each candidate dict.

        Args:
            candidates:         Ranked candidate dicts.
            job_description:    The job description text.
            enriched_contexts:  Optional mapping of candidate_name → enriched
                                context string (GitHub, portfolio, etc.).
                                A candidate's own 'enriched_context' field
                                takes precedence, since names can collide.
        """
        enriched_contexts = enriched_contexts or {}
        logger.info(f"Generating summaries for {len(candidates)} candidates")

        for candidate in candidates:
            name = candidate.get("candidate_name", "")
            enriched = candidate.get("enriched_context") or enriched_contexts.get(name, "")
            try:
                candidate["fit_summary"] = self.generate_fit_summary(
                    job_description=job_description,
                    resume_text=candidate["text"],
                    composite_score=candidate.get("composite_score", 0.0),
                    matching_skills=candidate.get("matching_skills"),
                    skill_coverage=candidate.get("skill_coverage_score", 0.0),
                    experience_score=candidate.get("experience_score", 0.0),
                    enriched_context=enriched,
                )
            except Exception as e:
                logger.error(f"Summary error for {name}: {e}")
                pct = candidate.get("percentage_score", 0.0)
                candidate["fit_summary"] = (
                    f"Candidate scored {pct:.1f}% overall match with this role."
                )

        return candidates

    # ------------------------------------------------------------------
    # Ollama-backed generation
    # ------------------------------------------------------------------

    def _generate_ollama_summary(
        self,
        job_description: str,
        resume_text: str,
        composite_score: float,
        matching_skills: list[str] | None,
        skill_coverage: float,
        experience_score: float,
        enriched_context: str = "",
    ) -> str:
        jd_snippet = job_description[:550]
        cv_snippet = resume_text[:1000]
        pct = composite_score * 100

        skills_line = (
            f"\nVerified matching skills: {', '.join(matching_skills[:12])}."
            if matching_skills
            else ""
        )
        enrichment_section = (
            f"\n\n--- Additional context from candidate's online presence ---\n{enriched_context[:900]}"
            if enriched_context
            else ""
        )

        prompt = f"""You are a senior technical recruiter writing a detailed, evidence-based candidate assessment report.

JOB DESCRIPTION:
{jd_snippet}

CANDIDATE RESUME:
{cv_snippet}{enrichment_section}

MATCH DATA:
- Overall score: {pct:.1f}%
- Skill coverage: {skill_coverage * 100:.0f}% of required skills found
- Experience alignment: {experience_score * 100:.0f}%{skills_line}

Write a 4–5 sentence assessment. Requirements:
1. Name specific technologies, companies, projects, or achievements from the candidate's background — do not speak in generalities
2. Explain precisely how their experience maps to the job requirements (what fits, what doesn't)
3. If GitHub or portfolio data was provided, cite specific repositories or projects that are relevant
4. Identify the single most important gap or question to probe in an interview (if any)
5. End with a hiring recommendation on its own line: "Recommendation: Strong Yes", "Recommendation: Yes", "Recommendation: Maybe", or "Recommendation: No" — followed by a single sentence explaining why

Do not use filler phrases like "strong candidate" or "great fit" unless you back them with specific evidence. Write in plain prose, no bullet points."""

        return self._call_ollama(prompt, num_predict=400)

    # ------------------------------------------------------------------
    # Template fallback (deterministic)
    # ------------------------------------------------------------------

    def _generate_template_summary(
        self,
        job_description: str,
        resume_text: str,
        composite_score: float,
        matching_skills: list[str] | None,
        skill_coverage: float,
        experience_score: float,
        enriched_context: str = "",
    ) -> str:
        """
        Build a factual summary from extracted signals.
        Output is deterministic — same inputs always produce the same summary.
        """
        cleaner = _text_cleaner

        if not matching_skills:
            job_skills = set(cleaner.extract_key_skills(job_description))
            resume_skills = set(cleaner.extract_key_skills(resume_text))
            matching_skills = sorted(job_skills & resume_skills)

        # Extract years of experience from resume
        years_matches = re.findall(r"(\d+)\+?\s*years?", resume_text, re.IGNORECASE)
        max_years = max((int(y) for y in years_matches), default=0)

        # Seniority signals
        seniority_words = [
            "senior",
            "lead",
            "principal",
            "staff",
            "architect",
            "manager",
            "director",
            "head of",
            "vp ",
            "vice president",
        ]
        is_senior = any(w in resume_text.lower() for w in seniority_words)

        # Education signals
        has_phd = bool(re.search(r"\bph\.?d\b|doctorate", resume_text, re.IGNORECASE))
        has_masters = bool(re.search(r"\bmaster'?s?\b|\bmsc\b|\bmba\b", resume_text, re.IGNORECASE))

        pct = composite_score * 100

        # --- Build sentences deterministically ---

        # Sentence 1: overall fit statement
        if pct >= 85:
            s1 = "This candidate is an excellent fit for the role."
        elif pct >= 70:
            s1 = "This candidate is a strong match for the role."
        elif pct >= 50:
            s1 = "This candidate meets the core requirements of the role."
        elif pct >= 25:
            s1 = "This candidate partially aligns with the role requirements."
        else:
            s1 = "This candidate has limited alignment with the role requirements."

        # Sentence 2: skills and experience
        parts = []
        if matching_skills:
            skills_str = ", ".join(matching_skills[:4])
            parts.append(f"matching skills in {skills_str}")
        if max_years > 0:
            level = "extensive" if max_years >= 8 else ("solid" if max_years >= 4 else "some")
            parts.append(f"{level} experience ({max_years}+ years)")
        elif is_senior:
            parts.append("senior-level background")
        if has_phd:
            parts.append("PhD-level education")
        elif has_masters:
            parts.append("advanced degree")

        if parts:
            s2 = "They bring " + ", and ".join(parts) + "."
        else:
            s2 = "Their background shows general technical competence."

        # Sentence 3: gap analysis or recommendation
        if pct >= 70:
            s3 = "Recommend for interview."
        elif pct >= 50:
            skill_gap = round((1.0 - skill_coverage) * 100)
            if skill_gap > 30:
                s3 = f"Around {skill_gap}% of required skills were not found — worth discussing in a screen."
            else:
                s3 = "Minor skill gaps; worth a screening conversation."
        elif pct >= 25:
            s3 = "Significant skill gaps exist; consider only if open to training investment."
        else:
            s3 = "Not recommended for this role."

        summary = f"{s1} {s2} {s3}"
        if enriched_context:
            summary += " Additional context from online profiles is available but requires Ollama for full analysis."
        return summary
