#!/bin/bash

# Setup script for Candidate Recommendation Engine (macOS compatible)
# Run this script to set up the project environment

echo "🚀 Setting up Candidate Recommendation Engine..."
echo "============================================="

# Check Python version (macOS compatible)
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version detected: $python_version"

# Simple version check
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
    echo "✅ Python version meets minimum requirement (3.10+)"
else
    echo "❌ Python 3.10+ is required. Please upgrade Python."
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -eq 0 ]; then
    echo "✅ Virtual environment created"
else
    echo "❌ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data/sample_resumes
mkdir -p logs
mkdir -p models/sentence_transformer

# Download models
echo ""
echo "Downloading AI models (this may take a few minutes)..."
python3 -c "
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import sys

print('Downloading embedding model...')
try:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print('✅ Embedding model downloaded')
except Exception as e:
    print(f'⚠️  Failed to download embedding model: {e}')
    print('You may need to download it when running the app')

print('Downloading summarization model...')
try:
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
    print('✅ Summarization model downloaded')
except Exception as e:
    print(f'⚠️  Failed to download summarization model: {e}')
    print('You may need to download it when running the app')
"

# Generate sample resumes
echo ""
echo "Generating sample resumes..."
if [ -f "generate_sample_resumes.py" ]; then
    python3 generate_sample_resumes.py
else
    echo "⚠️  generate_sample_resumes.py not found. Skipping sample generation."
fi

# Run tests (optional)
echo ""
echo "Running tests..."
if command -v pytest &> /dev/null; then
    pytest tests/ -v --tb=short || echo "⚠️  Some tests failed. This is expected if models are mocked."
else
    echo "⚠️  pytest not installed. Skipping tests."
fi

echo ""
echo "============================================="
echo "✅ Setup complete!"
echo ""
echo "To run the application:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the app: streamlit run src/app.py"
echo ""
echo "The app will be available at: http://localhost:8501"