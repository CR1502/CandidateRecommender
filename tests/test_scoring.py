"""
Unit tests for the composite-score components in EmbeddingEngine.
"""

from unittest.mock import Mock, patch

import pytest

from core.embeddings import EmbeddingEngine


@pytest.fixture(scope="module")
def engine():
    model = Mock()
    model.to.return_value = model
    with patch("core.embeddings.SentenceTransformer", return_value=model):
        yield EmbeddingEngine("test-model")


class TestExperienceSignal:
    JD = "We need 5+ years of Python experience."

    def test_no_requirement_is_neutral(self, engine):
        assert engine._experience_signal("Python developer wanted", "10 years") == 0.5

    def test_meets_requirement(self, engine):
        assert engine._experience_signal(self.JD, "7 years building APIs") == 1.0

    def test_monotonic_in_stated_years(self, engine):
        unstated = engine._experience_signal(self.JD, "Experienced engineer")
        scores = [engine._experience_signal(self.JD, f"{n} years") for n in range(1, 7)]
        assert unstated <= scores[0]
        assert scores == sorted(scores)

    def test_jd_range_uses_lower_bound(self, engine):
        jd = "3-5 years of experience required"
        assert engine._experience_signal(jd, "3 years") == 1.0

    def test_resume_range_uses_upper_bound(self, engine):
        assert engine._experience_signal(self.JD, "3 to 5 years") == 1.0

    def test_ignores_implausible_numbers(self, engine):
        resume = "Joined a company with 100 years of history. 2 years as analyst."
        assert engine._experience_signal(self.JD, resume) == pytest.approx(0.4)


class TestSkillCoverage:
    def test_full_coverage_with_many_resume_skills(self, engine):
        jd = "Requirements: Docker, Kubernetes, Terraform"
        resume = (
            "Python, Java, Kotlin, Scala, Ruby, PHP, JavaScript, TypeScript, HTML, "
            "CSS, Bash, React, Vue, Angular, Svelte, Docker, Kubernetes, Terraform"
        )
        assert engine._skill_coverage_score(jd, resume) == 1.0

    def test_partial_coverage(self, engine):
        assert engine._skill_coverage_score("C++ and Python", "C++ only") == 0.5

    def test_no_required_skills_is_neutral(self, engine):
        assert engine._skill_coverage_score("A friendly team player", "Python") == 0.5


def test_composite_is_weighted_and_clipped(engine):
    assert engine._composite_score(1.0, 1.0, 1.0) == pytest.approx(1.0)
    assert engine._composite_score(0.5, 0.0, 0.0) == pytest.approx(0.30)
    assert engine._composite_score(-1.0, 0.0, 0.0) == 0.0
