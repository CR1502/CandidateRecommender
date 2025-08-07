"""
AI-powered summarization for candidate fit explanations.
"""

from typing import Dict, Any, List, Optional
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from loguru import logger


class CandidateSummarizer:
    """Generate AI summaries explaining why candidates are good fits."""

    def __init__(self, model_name: str = "google/flan-t5-small"):
        """
        Initialize the summarizer with a local T5 model.

        Args:
            model_name: Name of the T5 model to use
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_length = 512
        self.summary_length = 100
        self._load_model()

    def _load_model(self) -> None:
        """Load the T5 model and tokenizer."""
        try:
            logger.info(f"Loading summarization model: {self.model_name}")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Summarization model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load summarization model: {e}")
            # Fall back to template-based summaries
            self.model = None
            self.tokenizer = None

    def generate_fit_summary(
            self,
            job_description: str,
            resume_text: str,
            similarity_score: float,
            matching_skills: Optional[List[str]] = None
    ) -> str:
        """
        Generate a summary explaining why the candidate is a good fit.

        Args:
            job_description: Job description text
            resume_text: Resume text
            similarity_score: Similarity score (0-1)
            matching_skills: Optional list of matching skills

        Returns:
            Generated summary text
        """
        # If model is available, use AI generation
        if self.model and self.tokenizer:
            try:
                return self._generate_ai_summary(
                    job_description,
                    resume_text,
                    similarity_score,
                    matching_skills
                )
            except Exception as e:
                logger.warning(f"AI summary generation failed: {e}")
                # Fall back to template

        # Use template-based summary as fallback
        return self._generate_template_summary(
            job_description,
            resume_text,
            similarity_score,
            matching_skills
        )

    def _generate_ai_summary(
            self,
            job_description: str,
            resume_text: str,
            similarity_score: float,
            matching_skills: Optional[List[str]] = None
    ) -> str:
        """
        Generate AI-powered summary using T5 model.

        Args:
            job_description: Job description text
            resume_text: Resume text
            similarity_score: Similarity score
            matching_skills: Optional matching skills

        Returns:
            AI-generated summary
        """
        # Truncate texts to fit model limits
        job_desc_truncated = job_description[:500]
        resume_truncated = resume_text[:500]

        # Create prompt
        prompt = f"""
        Task: Explain why this candidate is a good fit for the job in 2-3 sentences.

        Job requirements: {job_desc_truncated}

        Candidate background: {resume_truncated}

        Summary:
        """

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.summary_length,
                min_length=30,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                num_beams=2
            )

        # Decode output
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean up and enhance summary
        if matching_skills and len(matching_skills) > 0:
            skills_text = ", ".join(matching_skills[:3])
            summary += f" Key matching skills include {skills_text}."

        return summary.strip()

    def _generate_template_summary(
            self,
            job_description: str,
            resume_text: str,
            similarity_score: float,
            matching_skills: Optional[List[str]] = None
    ) -> str:
        """
        Generate template-based summary as fallback.

        Args:
            job_description: Job description text
            resume_text: Resume text
            similarity_score: Similarity score
            matching_skills: Optional matching skills

        Returns:
            Template-based summary
        """
        from .text_cleaner import TextCleaner
        cleaner = TextCleaner()

        # Extract skills if not provided
        if not matching_skills:
            job_skills = set(cleaner.extract_key_skills(job_description))
            resume_skills = set(cleaner.extract_key_skills(resume_text))
            matching_skills = list(job_skills & resume_skills)

        # Determine match level
        if similarity_score >= 0.8:
            match_level = "excellent"
            descriptor = "highly qualified"
        elif similarity_score >= 0.6:
            match_level = "strong"
            descriptor = "well-suited"
        elif similarity_score >= 0.4:
            match_level = "good"
            descriptor = "qualified"
        else:
            match_level = "potential"
            descriptor = "potentially suitable"

        # Build summary
        summary_parts = []

        # Main assessment
        summary_parts.append(
            f"This candidate shows {match_level} alignment with the position, "
            f"demonstrating {descriptor} expertise for the role."
        )

        # Skills mention
        if matching_skills:
            skills_text = ", ".join(matching_skills[:5])
            summary_parts.append(
                f"Strong competencies in {skills_text} directly match the job requirements."
            )

        # Experience indicator (based on resume length as proxy)
        if len(resume_text) > 2000:
            summary_parts.append(
                "The candidate's extensive background suggests significant relevant experience."
            )

        return " ".join(summary_parts[:2])  # Keep it concise

    def batch_generate_summaries(
            self,
            candidates: List[Dict[str, Any]],
            job_description: str
    ) -> List[Dict[str, Any]]:
        """
        Generate summaries for multiple candidates.

        Args:
            candidates: List of candidate dictionaries
            job_description: Job description text

        Returns:
            Updated candidates with summaries
        """
        logger.info(f"Generating summaries for {len(candidates)} candidates")

        for candidate in candidates:
            try:
                # Generate summary
                summary = self.generate_fit_summary(
                    job_description,
                    candidate['text'],
                    candidate['similarity_score'],
                    candidate.get('matching_skills')
                )
                candidate['fit_summary'] = summary

            except Exception as e:
                logger.error(f"Error generating summary for {candidate['candidate_name']}: {e}")
                candidate['fit_summary'] = (
                    f"Candidate shows {candidate['percentage_score']:.1f}% match "
                    f"with the job requirements based on resume analysis."
                )

        return candidates