"""
Unit tests for settings loading and validation.
"""

import pytest
from pydantic import ValidationError

from candidate_recommender.config import CANDIDATE_CATEGORIES, ScoringWeights, Settings


def test_defaults():
    settings = Settings(_env_file=None)
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.max_files_per_upload == 20


def test_env_overrides_use_the_old_variable_names(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    monkeypatch.setenv("OLLAMA_MODEL", "mistral")
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "5")
    monkeypatch.setenv(
        "SCORING_WEIGHTS", '{"semantic": 0.5, "skill_coverage": 0.4, "experience": 0.1}'
    )

    settings = Settings(_env_file=None)

    assert settings.embedding_model == "BAAI/bge-base-en-v1.5"
    assert settings.ollama_model == "mistral"
    assert settings.max_file_size_mb == 5
    assert settings.scoring_weights.skill_coverage == 0.4


def test_scoring_weights_must_sum_to_one():
    with pytest.raises(ValidationError, match="sum to 1.0"):
        ScoringWeights(semantic=0.7, skill_coverage=0.3, experience=0.1)


def test_categories_are_ordered_best_first_and_cover_zero():
    thresholds = [c.min_pct for c in CANDIDATE_CATEGORIES]
    assert thresholds == sorted(thresholds, reverse=True)
    assert thresholds[-1] == 0
