"""
Unit tests for the composite-score components in EmbeddingEngine.
"""

from unittest.mock import Mock, patch

import pytest

from candidate_recommender.core.embeddings import EmbeddingEngine, split_into_chunks


@pytest.fixture(scope="module")
def engine():
    model = Mock()
    model.to.return_value = model
    with patch("candidate_recommender.core.embeddings.SentenceTransformer", return_value=model):
        yield EmbeddingEngine("test-model")


class TestExperienceSignal:
    JD = "We need 5+ years of Python experience."

    def test_no_requirement_is_not_applicable(self, engine):
        assert engine._experience_signal("Python developer wanted", "10 years") is None

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

    def test_uses_employment_dates_when_years_not_stated(self, engine):
        resume = "Senior Engineer — Acme    Jan 2015 – Dec 2021\n- Built things"
        assert engine._experience_signal(self.JD, resume) == 1.0


class TestSkillCoverage:
    def test_described_skills_earn_full_credit(self, engine):
        jd = "Requirements: Docker, Kubernetes, Terraform"
        resume = "- Ran services on Kubernetes with Docker\n- Wrote Terraform modules"
        assert engine._skill_coverage_score(jd, resume) == 1.0

    def test_listed_only_skills_earn_partial_credit(self, engine):
        jd = "Requirements: Docker, Kubernetes, Terraform"
        resume = "Store Manager\nSKILLS\nPython, Java, React, Docker, Kubernetes, Terraform, AWS"
        assert engine._skill_coverage_score(jd, resume) == pytest.approx(0.5)

    def test_nice_to_have_skills_weigh_less(self, engine):
        jd = "Requirements: Python\nNice to have: Redis"
        assert engine._skill_coverage_score(jd, "Built Python services") == pytest.approx(1 / 1.5)
        assert engine._skill_coverage_score(jd, "Built Redis caches") == pytest.approx(0.5 / 1.5)

    def test_partial_coverage(self, engine):
        assert engine._skill_coverage_score("C++ and Python", "C++ only") == 0.5

    def test_no_required_skills_is_not_applicable(self, engine):
        assert engine._skill_coverage_score("A friendly team player", "Python") is None


class TestSemanticCalibration:
    def test_maps_working_range_onto_unit_interval(self, engine):
        floor, ceiling = engine.semantic_floor, engine.semantic_ceiling
        assert engine._calibrate_semantic(floor) == 0.0
        assert engine._calibrate_semantic(ceiling) == 1.0
        assert engine._calibrate_semantic((floor + ceiling) / 2) == pytest.approx(0.5)

    def test_clips_outside_range(self, engine):
        assert engine._calibrate_semantic(0.1) == 0.0
        assert engine._calibrate_semantic(0.99) == 1.0

    def test_rejects_inverted_range(self):
        with (
            patch("candidate_recommender.core.embeddings.SentenceTransformer"),
            pytest.raises(ValueError, match="ceiling"),
        ):
            EmbeddingEngine("m", semantic_floor=0.8, semantic_ceiling=0.5)


class TestCompositeScore:
    def test_all_components(self, engine):
        # relevance = (0.6*1 + 0.3*1)/0.9 = 1; score = (0.9*1 + 0.1*1*1)/1.0
        assert engine._composite_score(1.0, 1.0, 1.0) == pytest.approx(1.0)
        # relevance = 0.6*0.5/0.9 = 1/3; experience adds nothing when exp = 0
        assert engine._composite_score(0.5, 0.0, 0.0) == pytest.approx(0.9 * (1 / 3))

    def test_missing_components_are_renormalised_not_neutral(self, engine):
        # No skills or experience signal: score is the semantic match alone,
        # not inflated by a 0.5 "neutral" component.
        assert engine._composite_score(0.2, None, None) == pytest.approx(0.2)
        assert engine._composite_score(0.8, 0.4, None) == pytest.approx(
            (0.6 * 0.8 + 0.3 * 0.4) / 0.9
        )

    def test_experience_only_counts_when_relevant(self, engine):
        irrelevant_veteran = engine._composite_score(0.0, 0.0, 1.0)
        assert irrelevant_veteran == 0.0
        relevant_junior = engine._composite_score(0.9, 0.9, 0.3)
        relevant_senior = engine._composite_score(0.9, 0.9, 1.0)
        assert relevant_senior > relevant_junior

    def test_clipped(self, engine):
        assert engine._composite_score(-1.0, None, None) == 0.0


class TestChunking:
    def test_short_text_is_one_chunk(self):
        assert split_into_chunks("a b c", chunk_words=10, overlap_words=2) == ["a b c"]

    def test_disabled_with_zero(self):
        text = " ".join(str(i) for i in range(1000))
        assert split_into_chunks(text, chunk_words=0, overlap_words=50) == [text]

    def test_overlapping_windows_cover_everything(self):
        words = [str(i) for i in range(25)]
        chunks = split_into_chunks(" ".join(words), chunk_words=10, overlap_words=3)
        assert chunks[0].split() == words[:10]
        assert chunks[1].split()[0] == "7"  # 10 - 3 overlap
        assert chunks[-1].split()[-1] == "24"
        assert all(len(c.split()) <= 10 for c in chunks)

    def test_rejects_unknown_aggregation(self):
        with (
            patch("candidate_recommender.core.embeddings.SentenceTransformer"),
            pytest.raises(ValueError, match="chunk_aggregation"),
        ):
            EmbeddingEngine("m", chunk_aggregation="median")
