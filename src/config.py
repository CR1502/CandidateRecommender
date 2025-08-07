"""
Configuration settings for the Candidate Recommendation Engine.
"""

from pathlib import Path
from typing import Dict, List, Any
import os

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
SAMPLE_RESUMES_DIR = DATA_DIR / "sample_resumes"

# Model configurations
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARIZATION_MODEL_NAME = "google/flan-t5-small"

# File processing settings
ALLOWED_FILE_TYPES = ["pdf", "docx", "txt"]
MAX_FILE_SIZE_MB = 10
MAX_FILES_PER_UPLOAD = 20

# Processing settings
TOP_CANDIDATES_COUNT = 10
MIN_SIMILARITY_SCORE = 0.0  # Show all candidates
BATCH_SIZE = 32

# Text processing settings
MAX_TEXT_LENGTH = 10000  # Characters
MIN_TEXT_LENGTH = 50

# UI Settings
PAGE_TITLE = "Candidate Recommendation Engine"
PAGE_ICON = "🎯"
LAYOUT = "wide"

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "logs" / "app.log"

# Cache settings
ENABLE_CACHING = True
CACHE_TTL_SECONDS = 3600  # 1 hour

# Performance settings
MAX_WORKERS = 4
TIMEOUT_SECONDS = 30

# Prompt templates for summarization
SUMMARY_PROMPT_TEMPLATE = """
Given the job description and candidate resume, explain why this candidate is a good fit in 2-3 sentences.

Job Description: {job_description}

Candidate Resume: {resume_text}

Summary:
"""

# Sample data for testing
SAMPLE_JOB_DESCRIPTION = """
We are looking for a Senior Python Developer with machine learning experience.
Required skills:
- 5+ years of Python development
- Experience with ML frameworks (TensorFlow, PyTorch)
- Strong understanding of software engineering principles
- Experience with REST APIs and microservices
- Excellent problem-solving skills
"""

# Error messages
ERROR_MESSAGES = {
    "no_job_description": "Please enter a job description.",
    "no_files": "Please upload at least one resume file.",
    "file_too_large": "File size exceeds {max_size}MB limit.",
    "invalid_file_type": "Invalid file type. Allowed types: {allowed_types}",
    "processing_error": "Error processing file: {error}",
    "model_loading_error": "Error loading model: {error}",
}
