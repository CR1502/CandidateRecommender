"""
Text cleaning and preprocessing utilities.
"""

import re
from typing import Optional, List
from loguru import logger


class TextCleaner:
    """Clean and preprocess text for embedding generation."""

    def __init__(self):
        """Initialize the text cleaner."""
        self.min_length = 50
        self.max_length = 10000

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for processing.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text string

        Example:
            >>> cleaner = TextCleaner()
            >>> cleaned = cleaner.clean_text("  Some\\n\\ntext  with   spaces  ")
            >>> print(cleaned)
        """
        if not text:
            return ""

        try:
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)

            # Remove special characters but keep important punctuation
            text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)

            # Remove multiple consecutive punctuation marks
            text = re.sub(r'([\.,:;!?])\1+', r'\1', text)

            # Strip leading/trailing whitespace
            text = text.strip()

            # Truncate if too long
            if len(text) > self.max_length:
                logger.warning(f"Text truncated from {len(text)} to {self.max_length} characters")
                text = text[:self.max_length]

            return text

        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text

    def extract_candidate_name(self, text: str, filename: Optional[str] = None) -> str:
        """
        Extract candidate name from resume text or filename.

        Args:
            text: Resume text
            filename: Optional filename

        Returns:
            Extracted name or generated ID
        """
        # Try to extract from filename first
        if filename:
            # Remove extension and clean filename
            name = filename.rsplit('.', 1)[0]
            name = re.sub(r'[_\-]', ' ', name)
            name = re.sub(r'resume|cv|curriculum|vitae', '', name, flags=re.IGNORECASE)
            name = name.strip()
            if name and len(name) > 2:
                return name.title()

        # Try to extract from text (look for name patterns at the beginning)
        lines = text.split('\n')[:5]  # Check first 5 lines
        for line in lines:
            line = line.strip()
            # Simple heuristic: if line is 2-4 words and contains mostly letters
            words = line.split()
            if 2 <= len(words) <= 4:
                if all(word.replace('-', '').isalpha() for word in words):
                    return ' '.join(words).title()

        # Fallback to generated ID
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"Candidate_{text_hash}"

    def extract_key_skills(self, text: str) -> List[str]:
        """
        Extract key skills from text.

        Args:
            text: Resume or job description text

        Returns:
            List of extracted skills
        """
        # Common skill keywords (expandable)
        skill_patterns = [
            r'python', r'java\b', r'javascript', r'typescript', r'react', r'angular',
            r'node\.?js', r'django', r'flask', r'fastapi', r'sql', r'nosql', r'mongodb',
            r'postgresql', r'mysql', r'docker', r'kubernetes', r'aws', r'azure', r'gcp',
            r'machine learning', r'deep learning', r'tensorflow', r'pytorch', r'scikit-learn',
            r'pandas', r'numpy', r'data science', r'nlp', r'computer vision', r'ai\b',
            r'rest api', r'graphql', r'microservices', r'ci/cd', r'git', r'agile', r'scrum',
            r'leadership', r'communication', r'problem solving', r'teamwork'
        ]

        text_lower = text.lower()
        found_skills = []

        for pattern in skill_patterns:
            if re.search(pattern, text_lower):
                # Clean up the skill name
                skill = pattern.replace(r'\b', '').replace(r'\.?', '.').replace('\\', '')
                found_skills.append(skill.title())

        return list(set(found_skills))[:10]  # Return top 10 unique skills

    def prepare_for_embedding(self, text: str) -> str:
        """
        Prepare text specifically for embedding generation.

        Args:
            text: Text to prepare

        Returns:
            Prepared text for embedding
        """
        # Clean the text
        text = self.clean_text(text)

        # Remove very short text
        if len(text) < self.min_length:
            logger.warning(f"Text too short ({len(text)} chars), may affect quality")

        return text

    def validate_text(self, text: str) -> bool:
        """
        Validate if text is suitable for processing.

        Args:
            text: Text to validate

        Returns:
            True if valid, False otherwise
        """
        if not text or not text.strip():
            return False

        if len(text.strip()) < self.min_length:
            return False

        return True