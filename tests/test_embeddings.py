"""
Unit tests for embeddings module.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from candidate_recommender.core.embeddings import EmbeddingEngine


class TestEmbeddingEngine:
    """Test suite for EmbeddingEngine class."""

    @pytest.fixture(autouse=True)
    def _engine(self):
        """Set up an engine backed by a mocked SentenceTransformer."""
        self.mock_model = Mock()
        self.mock_model.encode.return_value = np.random.rand(384)
        self.mock_model.max_seq_length = 512
        self.mock_model.get_sentence_embedding_dimension.return_value = 384
        self.mock_model.to.return_value = self.mock_model

        with patch('candidate_recommender.core.embeddings.SentenceTransformer', return_value=self.mock_model):
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
        with pytest.raises(ValueError, match="All provided texts are empty"):
            self.engine.generate_embeddings_batch([])

    def test_generate_embeddings_batch_all_empty_texts(self):
        """Test batch embedding with all empty texts."""
        with pytest.raises(ValueError, match="All provided texts are empty"):
            self.engine.generate_embeddings_batch(["", " ", "\n"])

    def test_query_prefix_only_applied_to_queries(self):
        """BGE query prefix goes on the job description, not on resumes."""
        self.mock_model.encode.return_value = np.random.rand(384)
        self.engine.generate_embedding("Python developer", is_query=True)
        assert self.mock_model.encode.call_args[0][0].startswith(self.engine.query_prefix)

        self.engine.generate_embedding("Python developer", is_query=False)
        assert self.mock_model.encode.call_args[0][0] == "Python developer"

    def test_rank_candidates_orders_by_similarity(self):
        """With equal skill/experience signals, higher cosine ranks first."""
        job_emb = np.array([1.0, 0.0])
        resume_embs = np.array([[0.0, 1.0], [1.0, 0.0], [0.6, 0.8]])
        self.mock_model.encode.side_effect = [job_emb, resume_embs]
        resumes = [
            {"text": "Resume A", "candidate_name": "A", "filename": "a.txt"},
            {"text": "Resume B", "candidate_name": "B", "filename": "b.txt"},
            {"text": "Resume C", "candidate_name": "C", "filename": "c.txt"},
        ]

        ranked = self.engine.rank_candidates("Some job description", resumes)

        assert [r["candidate_name"] for r in ranked] == ["B", "C", "A"]
        assert [r["rank"] for r in ranked] == [1, 2, 3]

    def test_rank_candidates_removes_duplicates(self):
        self.mock_model.encode.side_effect = [np.array([1.0, 0.0]), np.array([[1.0, 0.0]])]
        resumes = [
            {"text": "Same resume", "candidate_name": "A", "filename": "a.txt"},
            {"text": "Same resume", "candidate_name": "B", "filename": "b.txt"},
        ]

        ranked = self.engine.rank_candidates("Some job description", resumes)

        assert [r["candidate_name"] for r in ranked] == ["A"]

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
        engine = EmbeddingEngine("BAAI/bge-small-en-v1.5")

        # Test embedding generation
        embedding = engine.generate_embedding("Test text")
        assert embedding.shape == (384,)

        # Embeddings are L2-normalised, so the dot product is cosine similarity
        job_emb = engine.generate_embedding("Python developer", is_query=True)
        resume_emb = engine.generate_embedding("Python programmer")
        assert float(np.dot(job_emb, resume_emb)) > 0.5
