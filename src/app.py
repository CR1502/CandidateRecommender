"""
Main Streamlit application for the Candidate Recommendation Engine.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import time
from typing import List, Dict, Any, Optional
from loguru import logger
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Import modules
from config import *
from core.file_processor import FileProcessor
from core.text_cleaner import TextCleaner
from core.embeddings import EmbeddingEngine
from core.summarizer import CandidateSummarizer

# Configure Streamlit page
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'job_description' not in st.session_state:
    st.session_state.job_description = ""
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []


@st.cache_resource
def load_models():
    """Load and cache ML models."""
    with st.spinner("Loading AI models... This may take a moment on first run."):
        try:
            embedding_engine = EmbeddingEngine(EMBEDDING_MODEL_NAME)
            summarizer = CandidateSummarizer(SUMMARIZATION_MODEL_NAME)
            return embedding_engine, summarizer
        except Exception as e:
            st.error(f"Failed to load models: {str(e)}")
            st.stop()


def process_candidates(
        job_description: str,
        uploaded_files: List[Any],
        embedding_engine: EmbeddingEngine,
        summarizer: CandidateSummarizer
) -> Dict[str, Any]:
    """
    Process candidates and generate recommendations.

    Args:
        job_description: Job description text
        uploaded_files: List of uploaded file objects
        embedding_engine: Embedding engine instance
        summarizer: Summarizer instance

    Returns:
        Dictionary with results
    """
    # Initialize processors
    file_processor = FileProcessor(MAX_FILE_SIZE_MB)
    text_cleaner = TextCleaner()

    # Process files
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Extract text from files
    status_text.text("📄 Processing resume files...")
    processed_resumes = []

    for i, file in enumerate(uploaded_files):
        progress_bar.progress((i + 1) / len(uploaded_files) * 0.3)

        # Validate file
        is_valid, error_msg = file_processor.validate_file(file, file.name)
        if not is_valid:
            st.warning(f"⚠️ {file.name}: {error_msg}")
            continue

        try:
            # Process file
            text, candidate_name = file_processor.process_file(file, file.name)

            # Clean text
            cleaned_text = text_cleaner.prepare_for_embedding(text)

            if text_cleaner.validate_text(cleaned_text):
                processed_resumes.append({
                    'filename': file.name,
                    'candidate_name': candidate_name,
                    'text': cleaned_text
                })
            else:
                st.warning(f"⚠️ {file.name}: Text too short or invalid")

        except Exception as e:
            st.warning(f"⚠️ Error processing {file.name}: {str(e)}")

    if not processed_resumes:
        st.error("No valid resumes could be processed.")
        return None

    # Step 2: Rank candidates
    progress_bar.progress(0.5)
    status_text.text("🤖 Analyzing candidates with AI...")

    try:
        # Clean job description
        cleaned_job_desc = text_cleaner.prepare_for_embedding(job_description)

        # Rank candidates
        ranked_candidates = embedding_engine.rank_candidates(
            cleaned_job_desc,
            processed_resumes,
            TOP_CANDIDATES_COUNT
        )

        # Extract matching skills for each candidate
        for candidate in ranked_candidates:
            candidate['matching_skills'] = embedding_engine.find_matching_skills(
                cleaned_job_desc,
                candidate['text']
            )

    except Exception as e:
        st.error(f"Error ranking candidates: {str(e)}")
        return None

    # Step 3: Generate summaries
    progress_bar.progress(0.8)
    status_text.text("✍️ Generating fit summaries...")

    try:
        ranked_candidates = summarizer.batch_generate_summaries(
            ranked_candidates,
            cleaned_job_desc
        )
    except Exception as e:
        logger.warning(f"Summary generation failed: {e}")
        # Continue without summaries

    # Complete
    progress_bar.progress(1.0)
    status_text.text("✅ Analysis complete!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

    return {
        'candidates': ranked_candidates,
        'total_processed': len(processed_resumes),
        'job_description': job_description
    }


def display_results(results: Dict[str, Any]) -> None:
    """
    Display recommendation results.

    Args:
        results: Results dictionary from process_candidates
    """
    st.success(f"✅ Analyzed {results['total_processed']} resumes successfully!")

    # Display top candidates
    st.markdown("## 🏆 Top Candidate Recommendations")

    candidates = results['candidates']

    for i, candidate in enumerate(candidates):
        # Create expandable card for each candidate
        with st.expander(
                f"**#{candidate['rank']} - {candidate['candidate_name']}** "
                f"(Match: {candidate['percentage_score']:.1f}%)",
                expanded=(i < 3)  # Expand top 3 by default
        ):
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Match Score", f"{candidate['percentage_score']:.1f}%")
            with col2:
                st.metric("Rank", f"#{candidate['rank']}")
            with col3:
                st.metric("Source", candidate['filename'][:20] + "...")

            # Display fit summary
            st.markdown("### 📝 Why This Candidate is a Great Fit")
            st.info(candidate.get('fit_summary', 'Summary not available'))

            # Display matching skills
            if candidate.get('matching_skills'):
                st.markdown("### 🎯 Matching Skills")
                skills_cols = st.columns(5)
                for idx, skill in enumerate(candidate['matching_skills'][:10]):
                    with skills_cols[idx % 5]:
                        st.badge(skill)

            # Add action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"📧 Contact", key=f"contact_{i}"):
                    st.info("Contact feature coming soon!")
            with col2:
                if st.button(f"📄 View Full Resume", key=f"view_{i}"):
                    with st.container():
                        st.text_area(
                            "Resume Content",
                            candidate['text'][:1000] + "...",
                            height=200,
                            key=f"resume_text_{i}"
                        )


def export_results(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Export results to DataFrame.

    Args:
        results: Results dictionary

    Returns:
        DataFrame with candidate data
    """
    data = []
    for candidate in results['candidates']:
        data.append({
            'Rank': candidate['rank'],
            'Candidate Name': candidate['candidate_name'],
            'Match Score (%)': round(candidate['percentage_score'], 2),
            'Source File': candidate['filename'],
            'Matching Skills': ', '.join(candidate.get('matching_skills', [])),
            'Fit Summary': candidate.get('fit_summary', '')
        })

    return pd.DataFrame(data)


def main():
    """Main application function."""

    # Header
    st.title(PAGE_TITLE)
    st.markdown(
        "Upload resumes and enter a job description to find the best candidates using AI-powered matching."
    )

    # Load models
    embedding_engine, summarizer = load_models()

    # Sidebar for configuration
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        # Model info
        with st.expander("Model Information"):
            model_info = embedding_engine.get_model_info()
            for key, value in model_info.items():
                st.text(f"{key}: {value}")

        # Sample data
        if st.button("Load Sample Job Description"):
            st.session_state.job_description = SAMPLE_JOB_DESCRIPTION
            st.rerun()

        # Help section
        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            1. **Enter Job Description**: Paste or type the job requirements
            2. **Upload Resumes**: Select PDF, DOCX, or TXT files (max 10MB each)
            3. **Find Candidates**: Click to analyze and rank candidates
            4. **Review Results**: Explore top matches with AI-generated insights
            5. **Export**: Download results as CSV for further review
            """)

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📋 Job Description")
        job_description = st.text_area(
            "Enter the job description and requirements:",
            value=st.session_state.job_description,
            height=300,
            placeholder="e.g., We are looking for a Senior Python Developer with 5+ years of experience..."
        )
        st.session_state.job_description = job_description

    with col2:
        st.markdown("### 📁 Resume Upload")
        uploaded_files = st.file_uploader(
            "Select resume files to analyze:",
            type=ALLOWED_FILE_TYPES,
            accept_multiple_files=True,
            help=f"Supported formats: {', '.join(ALLOWED_FILE_TYPES)}. Max {MAX_FILE_SIZE_MB}MB per file."
        )

        if uploaded_files:
            st.info(f"📎 {len(uploaded_files)} file(s) uploaded")

            # Show uploaded files
            with st.expander("View uploaded files"):
                for file in uploaded_files:
                    file_size_mb = len(file.getvalue()) / (1024 * 1024)
                    st.text(f"• {file.name} ({file_size_mb:.2f} MB)")

    # Action buttons
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button("🔍 Find Best Candidates", type="primary", use_container_width=True):
            # Validate inputs
            if not job_description or not job_description.strip():
                st.error(ERROR_MESSAGES['no_job_description'])
            elif not uploaded_files:
                st.error(ERROR_MESSAGES['no_files'])
            else:
                # Process candidates
                with st.spinner("Analyzing candidates..."):
                    results = process_candidates(
                        job_description,
                        uploaded_files,
                        embedding_engine,
                        summarizer
                    )

                    if results:
                        st.session_state.results = results
                        st.rerun()

    with col2:
        if st.button("🔄 Clear All", use_container_width=True):
            st.session_state.results = None
            st.session_state.job_description = ""
            st.rerun()

    # Display results if available
    if st.session_state.results:
        st.markdown("---")
        display_results(st.session_state.results)

        # Export options
        st.markdown("---")
        st.markdown("### 📊 Export Results")

        col1, col2 = st.columns(2)

        with col1:
            # Export as CSV
            df = export_results(st.session_state.results)
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="candidate_recommendations.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            # Display as table
            if st.button("📋 View as Table", use_container_width=True):
                st.dataframe(df, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ using Streamlit and Local AI Models</p>
            <p>All processing happens locally - No external APIs required</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    # Setup logging
    logger.add(
        LOG_FILE,
        rotation="10 MB",
        level=LOG_LEVEL
    )

    # Run app
    main()