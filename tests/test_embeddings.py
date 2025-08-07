"""
Unit tests for embeddings module.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.embeddings import EmbeddingEngine


class TestEmbeddingEngine:
    """Test suite for EmbeddingEngine class."""

    @patch('core.embeddings.SentenceTransformer')
    def setup_method(self, mock_transformer):
        """Set up test fixtures with mocked model."""
        # Mock the model
        self.mock_model = Mock()
        self.mock_model.encode.return_value = np.random.rand(384)
        self.mock_model.max_seq_length = 512
        self.mock_model.get_sentence_embedding_dimension.return_value = 384
        self.mock_model.to.return_value = self.mock_model

        mock_transformer.return_value = self.mock_model

        self.engine = EmbeddingEngine("test-model")

    def test_init(self):
        """Test EmbeddingEngine initialization."""
        assert self.engine.model_name == "test-model"
        assert self.engine.model is not None

    def test_generate_embedding(self):
        """Test single embedding generation."""
        text = "Python developer with 5 years experience"

        # Mock return value
        expected_embedding = np.random.rand(384)
        self.mock_model.encode.return_value = expected_embedding

        embedding = self.engine.generate_embedding(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        self.mock_model.encode.assert_called_once()

    def test_generate_embedding_empty_text(self):
        """Test embedding generation with empty text."""
        with pytest.raises(ValueError, match="Input text is empty"):
            self.engine.generate_embedding("")

    def test_generate_embeddings_batch(self):
        """Test batch embedding generation."""
        texts = [
            "Python developer",
            "Java engineer",
            "Data scientist"
        ]

        # Mock return value
        expected_embeddings = np.random.rand(3, 384)
        self.mock_model.encode.return_value = expected_embeddings

        embeddings = self.engine.generate_embeddings_batch(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
        self.mock_model.encode.assert_called_once()

    def test_generate_embeddings_batch_empty_list(self):
        """Test batch embedding with empty list."""
        with pytest.raises(ValueError, match="No texts provided"):
            self.engine.generate_embeddings_batch([])

    def test_generate_embeddings_batch_all_empty_texts(self):
        """Test batch embedding with all empty texts."""
        with pytest.raises(ValueError, match="All texts are empty"):
            self.engine.generate_embeddings_batch(["", " ", "\n"])

    def test_compute_similarity(self):
        """Test similarity computation."""
        job_embedding = np.random.rand(384)
        resume_embeddings = np.random.rand(5, 384)

        similarities = self.engine.compute_similarity(job_embedding, resume_embeddings)

        assert isinstance(similarities, np.ndarray)
        assert similarities.shape == (5,)
        assert all(0 <= s <= 1 for s in similarities)

    def test_compute_similarity_single_resume(self):
        """Test similarity with single resume."""
        job_embedding = np.random.rand(384)
        resume_embedding = np.random.rand(384)

        similarity = self.engine.compute_similarity(job_embedding, resume_embedding)

        assert isinstance(similarity, np.ndarray)
        assert similarity.shape == (1,)

    def test_rank_candidates(self):
        """Test candidate ranking."""
        job_description = "Python developer with ML experience"
        resumes = [
            {"text": "Python expert", "candidate_name": "John"},
            {"text": "Java developer", "candidate_name": "Jane"},
            {"text": "ML engineer", "candidate_name": "Bob"}
        ]

        # Mock embeddings
        job_emb = np.random.rand(384)
        resume_embs = np.random.rand(3, 384)

        self.mock_model.encode.side_effect = [job_emb, resume_embs]

        ranked = self.engine.rank_candidates(job_description, resumes, top_k=2)

        assert len(ranked) == 2
        assert all('similarity_score' in r for r in ranked)
        assert all('percentage_score' in r for r in ranked)
        assert all('rank' in r for r in ranked)
        assert ranked[0]['rank'] == 1
        assert ranked[1]['rank'] == 2

    def test_rank_candidates_empty_job_description(self):
        """Test ranking with empty job description."""
        with pytest.raises(ValueError, match="Job description is empty"):
            self.engine.rank_candidates("", [{"text": "Resume"}], top_k=5)

    def test_rank_candidates_no_resumes(self):
        """Test ranking with no resumes."""
        with pytest.raises(ValueError, match="No resumes provided"):
            self.engine.rank_candidates("Job description", [], top_k=5)

    def test_find_matching_skills(self):
        """Test skill matching between job and resume."""
        job_text = "Python developer with Django and PostgreSQL experience"
        resume_text = "Experienced in Python, Django, MySQL, and React"

        skills = self.engine.find_matching_skills(job_text, resume_text)

        assert isinstance(skills, list)
        assert "Python" in skills
        assert "Django" in skills
        assert "PostgreSQL" not in skills  # Not in resume
        assert "React" not in skills  # Not in job description

    def test_get_model_info(self):
        """Test getting model information."""
        info = self.engine.get_model_info()

        assert 'model_name' in info
        assert 'device' in info
        assert 'max_seq_length' in info
        assert 'embedding_dimension' in info
        assert info['model_name'] == "test-model"
        assert info['embedding_dimension'] == 384


class TestEmbeddingEngineIntegration:
    """Integration tests for EmbeddingEngine."""

    @pytest.mark.skipif(
        True,  # Skip by default as it requires downloading models
        reason="Requires downloading actual models"
    )
    def test_real_model_loading(self):
        """Test with actual model loading."""
        engine = EmbeddingEngine("sentence-transformers/all-MiniLM-L6-v2")

        # Test embedding generation
        embedding = engine.generate_embedding("Test text")
        assert embedding.shape == (384,)

        # Test similarity
        job_emb = engine.generate_embedding("Python developer")
        resume_emb = engine.generate_embedding("Python programmer")

        similarity = engine.compute_similarity(job_emb, resume_emb)
        assert similarity[0] > 0.5  # Should be reasonably similar