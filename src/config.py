"""
Configuration settings for the Candidate Recommendation Engine.
"""

from pathlib import Path
import os

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
SAMPLE_RESUMES_DIR = DATA_DIR / "sample_resumes"

# Embedding model — bge-small is a significant step up from MiniLM with minimal size increase.
# For highest accuracy use BAAI/bge-base-en-v1.5 (~420MB). bge-small is ~130MB.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# BGE models improve retrieval when queries have this prefix (passages/resumes don't need it)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Ollama settings — free local LLM inference. Install from https://ollama.com
# Run: ollama pull llama3.2 (3B, fast) or ollama pull mistral (7B, higher quality)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))

# Composite scoring weights — must sum to 1.0
# semantic: overall semantic match via embeddings
# skill_coverage: fraction of required skills found in the resume
# experience: whether stated years of experience meets the job requirement
SCORING_WEIGHTS = {
    "semantic": 0.60,
    "skill_coverage": 0.30,
    "experience": 0.10,
}

# File processing settings
ALLOWED_FILE_TYPES = ["pdf", "docx", "txt"]
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILES_PER_UPLOAD = int(os.getenv("MAX_FILES_PER_UPLOAD", "20"))

# Processing settings
TOP_CANDIDATES_COUNT = int(os.getenv("TOP_CANDIDATES_COUNT", "10"))
MIN_SIMILARITY_SCORE = 0.15  # Hide candidates below this threshold
BATCH_SIZE = 32

# Text processing settings
MAX_TEXT_LENGTH = 12000
MIN_TEXT_LENGTH = 50

# Candidate categories (thresholds are on the composite 0–100 scale)
CANDIDATE_CATEGORIES = {
    "perfect":         {"min": 85, "max": 100, "label": "Perfect Match",    "emoji": "🌟", "color": "#00D26A"},
    "ideal":           {"min": 70, "max": 85,  "label": "Ideal Candidate",  "emoji": "⭐", "color": "#4CAF50"},
    "good":            {"min": 50, "max": 70,  "label": "Good Candidate",   "emoji": "✅", "color": "#FFA726"},
    "okay":            {"min": 25, "max": 50,  "label": "Okay Candidate",   "emoji": "👍", "color": "#FF9800"},
    "not_recommended": {"min": 0,  "max": 25,  "label": "Not Recommended",  "emoji": "❌", "color": "#F44336"},
}

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "logs" / "app.log"

# Cache
ENABLE_CACHING = True
CACHE_TTL_SECONDS = 3600

# Sample job description for testing
SAMPLE_JOB_DESCRIPTION = """
We are looking for a Senior Python Developer with machine learning experience.

Required skills:
- 5+ years of Python development
- Experience with ML frameworks (TensorFlow, PyTorch)
- Strong understanding of software engineering principles
- Experience with REST APIs and microservices
- Docker and Kubernetes experience
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
