"""
Embedding generation and similarity computation utilities.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
import torch


class EmbeddingEngine:
    """Generate embeddings and compute similarities using local models."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding engine with a local model.

        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load_model()

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load embedding model: {str(e)}")

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector as numpy array

        Example:
            >>> engine = EmbeddingEngine()
            >>> embedding = engine.generate_embedding("Python developer with ML experience")
            >>> print(embedding.shape)
        """
        if not text or not text.strip():
            raise ValueError("Input text is empty")

        try:
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batch.

        Args:
            texts: List of input texts

        Returns:
            Array of embedding vectors
        """
        if not texts:
            raise ValueError("No texts provided")

        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("All texts are empty")

        try:
            logger.info(f"Generating embeddings for {len(valid_texts)} texts")

            # Generate embeddings in batch
            embeddings = self.model.encode(
                valid_texts,
                convert_to_numpy=True,
                show_progress_bar=len(valid_texts) > 10,
                batch_size=32
            )

            return embeddings

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise

    def compute_similarity(
            self,
            job_embedding: np.ndarray,
            resume_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between job description and resumes.

        Args:
            job_embedding: Embedding vector for job description
            resume_embeddings: Array of embedding vectors for resumes

        Returns:
            Array of similarity scores (0-1 range)
        """
        try:
            # Reshape job embedding if needed
            if len(job_embedding.shape) == 1:
                job_embedding = job_embedding.reshape(1, -1)

            # Ensure resume embeddings is 2D
            if len(resume_embeddings.shape) == 1:
                resume_embeddings = resume_embeddings.reshape(1, -1)

            # Compute cosine similarity
            similarities = cosine_similarity(job_embedding, resume_embeddings)

            # Return flattened array
            return similarities.flatten()

        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            raise

    def rank_candidates(
            self,
            job_description: str,
            resumes: List[Dict[str, Any]],
            top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rank candidates based on similarity to job description.

        Args:
            job_description: Job description text
            resumes: List of resume dictionaries with 'text' and 'candidate_name'
            top_k: Number of top candidates to return

        Returns:
            List of ranked candidates with similarity scores
        """
        if not job_description:
            raise ValueError("Job description is empty")

        if not resumes:
            raise ValueError("No resumes provided")

        try:
            # Generate job embedding
            logger.info("Generating job description embedding")
            job_embedding = self.generate_embedding(job_description)

            # Extract resume texts
            resume_texts = [r['text'] for r in resumes]

            # Generate resume embeddings
            logger.info(f"Generating embeddings for {len(resume_texts)} resumes")
            resume_embeddings = self.generate_embeddings_batch(resume_texts)

            # Compute similarities
            similarities = self.compute_similarity(job_embedding, resume_embeddings)

            # Create results with scores
            results = []
            for i, (resume, score) in enumerate(zip(resumes, similarities)):
                results.append({
                    'candidate_name': resume.get('candidate_name', f'Candidate {i + 1}'),
                    'filename': resume.get('filename', ''),
                    'text': resume['text'],
                    'similarity_score': float(score),
                    'percentage_score': float(score * 100),
                    'rank': 0  # Will be set after sorting
                })

            # Sort by similarity score (descending)
            results.sort(key=lambda x: x['similarity_score'], reverse=True)

            # Add ranks
            for i, result in enumerate(results):
                result['rank'] = i + 1

            # Return top k results
            return results[:top_k]

        except Exception as e:
            logger.error(f"Error ranking candidates: {e}")
            raise

    def find_matching_skills(
            self,
            job_text: str,
            resume_text: str
    ) -> List[str]:
        """
        Find skills that match between job description and resume.

        Args:
            job_text: Job description text
            resume_text: Resume text

        Returns:
            List of matching skills
        """
        from .text_cleaner import TextCleaner

        cleaner = TextCleaner()
        job_skills = set(cleaner.extract_key_skills(job_text))
        resume_skills = set(cleaner.extract_key_skills(resume_text))

        # Find intersection
        matching_skills = list(job_skills & resume_skills)

        return matching_skills

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if not self.model:
            return {"error": "Model not loaded"}

        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_seq_length": self.model.max_seq_length,
            "embedding_dimension": self.model.get_sentence_embedding_dimension()
        }