"""
Unit tests for the template (non-LLM) summary, including components that
don't apply to the job (None).
"""

from unittest.mock import patch

import pytest

from candidate_recommender.core.summarizer import CandidateSummarizer


@pytest.fixture
def summarizer():
    with patch.object(CandidateSummarizer, "_check_ollama", return_value=False):
        return CandidateSummarizer()


@pytest.mark.parametrize("score", [0.9, 0.6, 0.3, 0.1])
def test_template_handles_not_applicable_components(summarizer, score):
    summary = summarizer.generate_fit_summary(
        job_description="Digital marketing manager, SEO and paid social",
        resume_text="Marketing manager with Google Ads and SEO experience",
        composite_score=score,
        skill_coverage=None,
        experience_score=None,
    )
    assert summary and "None" not in summary


def test_template_mentions_skill_gap_when_coverage_is_low(summarizer):
    summary = summarizer.generate_fit_summary(
        job_description="Python, Docker, Kubernetes",
        resume_text="Python developer",
        composite_score=0.6,
        matching_skills=["Python"],
        skill_coverage=0.3,
        experience_score=1.0,
    )
    assert "70% of required skills were not found" in summary


def test_llm_prompt_marks_not_applicable_components(summarizer):
    with patch.object(summarizer, "_call_ollama", return_value="ok") as call:
        summarizer._generate_ollama_summary("JD", "resume", 0.5, [], None, None)
    prompt = call.call_args[0][0]
    assert "Skill coverage: not applicable" in prompt
    assert "Experience alignment: not applicable" in prompt


def test_template_years_ignore_percentages(summarizer):
    # "32% year over year" once read as "32 years" after cleaning dropped the "%".
    resume = (
        "Marketing Manager — Glow    2021 – Present\n"
        "- ROAS up 32% year over year\n"
        "7 years of experience in digital marketing"
    )
    candidate = {"text": "cleaned", "raw_text": resume, "composite_score": 0.8}
    [result] = summarizer.batch_generate_summaries([candidate], "Marketing manager")
    assert "32" not in result["fit_summary"]
    assert "7+ years" in result["fit_summary"]
