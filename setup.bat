@echo off
REM Setup script for Candidate Recommendation Engine (Windows)
REM Run this script to set up the project environment

echo Setting up Candidate Recommendation Engine...
echo =============================================

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.10 or higher
    pause
    exit /b 1
)

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo Error: Failed to create virtual environment
    pause
    exit /b 1
)
echo Virtual environment created

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

REM Create necessary directories
echo.
echo Creating project directories...
if not exist "data\sample_resumes" mkdir data\sample_resumes
if not exist "logs" mkdir logs
if not exist "models\sentence_transformer" mkdir models\sentence_transformer

REM Download models
echo.
echo Downloading AI models (this may take a few minutes)...
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('Embedding model downloaded')"
python -c "from transformers import T5ForConditionalGeneration, T5Tokenizer; tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small'); model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small'); print('Summarization model downloaded')"

REM Generate sample resumes
echo.
echo Generating sample resumes...
python generate_sample_resumes.py

REM Run tests
echo.
echo Running tests...
pytest tests\ -v --tb=short

echo.
echo =============================================
echo Setup complete!
echo.
echo To run the application:
echo 1. Activate the virtual environment: venv\Scripts\activate
echo 2. Run the app: streamlit run src\app.py
echo.
echo The app will be available at: http://localhost:8501
echo.
pause