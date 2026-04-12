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
            embedding_engine = EmbeddingEngine(
                model_name=EMBEDDING_MODEL_NAME,
                query_prefix=BGE_QUERY_PREFIX,
                scoring_weights=SCORING_WEIGHTS,
            )
            summarizer = CandidateSummarizer(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                timeout=OLLAMA_TIMEOUT,
            )
            return embedding_engine, summarizer
        except Exception as e:
            st.error(f"Failed to load models: {str(e)}")
            st.stop()


def extract_contact_info(resume_text: str) -> Dict[str, Optional[str]]:
    """Extract contact information from raw (uncleaned) resume text."""
    return TextCleaner().extract_contact_details(resume_text)


def validate_text_content(text: str, min_length: int = 50) -> bool:
    """
    Validate if text is suitable for processing.
    
    Args:
        text: Text to validate
        min_length: Minimum text length required
        
    Returns:
        True if valid, False otherwise
    """
    if not text or not text.strip():
        return False
    
    if len(text.strip()) < min_length:
        return False
    
    return True


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
    
    # Try to use TextCleaner, but have fallbacks for all methods
    try:
        text_cleaner = TextCleaner()
    except:
        text_cleaner = None
    
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
            
            # Clean text - with fallback
            if text_cleaner and hasattr(text_cleaner, 'prepare_for_embedding'):
                cleaned_text = text_cleaner.prepare_for_embedding(text)
            else:
                # Fallback cleaning
                import re
                cleaned_text = re.sub(r'\s+', ' ', text).strip()
                if len(cleaned_text) > 10000:
                    cleaned_text = cleaned_text[:10000]
            
            # Validate text using inline function
            if validate_text_content(cleaned_text):
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
        # Clean job description - with fallback
        if text_cleaner and hasattr(text_cleaner, 'prepare_for_embedding'):
            cleaned_job_desc = text_cleaner.prepare_for_embedding(job_description)
        else:
            # Fallback cleaning
            import re
            cleaned_job_desc = re.sub(r'\s+', ' ', job_description).strip()
        
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
    Display recommendation results with category classifications.
    
    Args:
        results: Results dictionary from process_candidates
    """
    st.success(f"✅ Analyzed {results['total_processed']} resumes successfully!")
    
    # Display top candidates
    st.markdown("## 🏆 Candidate Recommendations")
    
    candidates = results['candidates']
    
    # Ensure all candidates have category fields (for backward compatibility)
    for candidate in candidates:
        if 'category' not in candidate or 'category_emoji' not in candidate:
            percentage = candidate['percentage_score']
            if percentage >= 90:
                candidate['category'] = "Perfect Match"
                candidate['category_emoji'] = "🌟"
                candidate['category_color'] = "#00D26A"
            elif percentage >= 70:
                candidate['category'] = "Ideal Candidate"
                candidate['category_emoji'] = "⭐"
                candidate['category_color'] = "#4CAF50"
            elif percentage >= 50:
                candidate['category'] = "Good Candidate"
                candidate['category_emoji'] = "✅"
                candidate['category_color'] = "#FFA726"
            elif percentage >= 20:
                candidate['category'] = "Okay Candidate"
                candidate['category_emoji'] = "👍"
                candidate['category_color'] = "#FF9800"
            else:
                candidate['category'] = "Not Recommended"
                candidate['category_emoji'] = "❌"
                candidate['category_color'] = "#F44336"
    
    # Filter out "Not Recommended" candidates unless explicitly shown
    show_not_recommended = st.checkbox("Show 'Not Recommended' candidates (below 20%)", value=False)
    
    if not show_not_recommended:
        candidates = [c for c in candidates if c['percentage_score'] >= 20]
    
    if not candidates:
        st.warning("No candidates meet the minimum threshold. Try adjusting your requirements or check 'Show Not Recommended' above.")
        return
    
    # Group candidates by category
    perfect_matches = [c for c in candidates if c['percentage_score'] >= 90]
    ideal_candidates = [c for c in candidates if 70 <= c['percentage_score'] < 90]
    good_candidates = [c for c in candidates if 50 <= c['percentage_score'] < 70]
    okay_candidates = [c for c in candidates if 20 <= c['percentage_score'] < 50]
    not_recommended = [c for c in candidates if c['percentage_score'] < 20]
    
    # Display category summary
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("🌟 Perfect", len(perfect_matches))
    with col2:
        st.metric("⭐ Ideal", len(ideal_candidates))
    with col3:
        st.metric("✅ Good", len(good_candidates))
    with col4:
        st.metric("👍 Okay", len(okay_candidates))
    with col5:
        if show_not_recommended:
            st.metric("❌ Not Rec.", len(not_recommended))
    
    st.markdown("---")
    
    # Display candidates by category
    for category_name, category_candidates, expanded_default in [
        ("🌟 Perfect Matches (90-100%)", perfect_matches, True),
        ("⭐ Ideal Candidates (70-90%)", ideal_candidates, True),
        ("✅ Good Candidates (50-70%)", good_candidates, False),
        ("👍 Okay Candidates (20-50%)", okay_candidates, False),
        ("❌ Not Recommended (<20%)", not_recommended, False)
    ]:
        if category_candidates and (show_not_recommended or "Not Recommended" not in category_name):
            st.markdown(f"### {category_name}")
            
            for i, candidate in enumerate(category_candidates):
                # Color-coded expander based on category
                category_color = candidate.get('category_color', '#808080')
                category_emoji = candidate.get('category_emoji', '📋')
                category = candidate.get('category', 'Candidate')
                
                with st.expander(
                    f"{category_emoji} **{candidate['candidate_name']}** "
                    f"(Match: {candidate['percentage_score']:.1f}%)",
                    expanded=(expanded_default and i < 2)  # Expand first 2 in top categories
                ):
                    # Display metrics with color coding
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Match Score", f"{candidate['percentage_score']:.1f}%")
                    with col2:
                        st.metric("Category", category)
                    with col3:
                        st.metric("Rank", f"#{candidate['rank']}")
                    with col4:
                        st.metric("Source", candidate['filename'][:15] + "...")
                    
                    # Category-specific styling for summary
                    st.markdown("### 📝 Assessment")
                    
                    # Color-code the info box based on category
                    if candidate['percentage_score'] >= 90:
                        st.success(candidate.get('fit_summary', 'Summary not available'))
                    elif candidate['percentage_score'] >= 70:
                        st.info(candidate.get('fit_summary', 'Summary not available'))
                    elif candidate['percentage_score'] >= 50:
                        st.warning(candidate.get('fit_summary', 'Summary not available'))
                    else:
                        st.error(candidate.get('fit_summary', 'Summary not available'))
                    
                    # Display matching skills
                    if candidate.get('matching_skills'):
                        st.markdown("### 🎯 Matching Skills")
                        skills_cols = st.columns(5)
                        for idx, skill in enumerate(candidate['matching_skills'][:10]):
                            with skills_cols[idx % 5]:
                                st.badge(skill)
                    
                    # Add action buttons based on category
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"📋 Get Info", key=f"info_{candidate['candidate_name']}_{i}", type="primary" if candidate['percentage_score'] >= 70 else "secondary"):
                            # Extract contact info from resume
                            contact_info = extract_contact_info(candidate['text'])
                            
                            # Display in a nice formatted box
                            with st.container():
                                st.markdown("#### 📇 Contact Information")
                                
                                contact_cols = st.columns(2)
                                
                                with contact_cols[0]:
                                    if contact_info['email']:
                                        st.markdown(f"**📧 Email**")
                                        st.code(contact_info['email'], language=None)
                                    else:
                                        st.markdown("**📧 Email**")
                                        st.text("Not found")
                                    
                                    if contact_info['phone']:
                                        st.markdown(f"**📱 Phone**")
                                        st.code(contact_info['phone'], language=None)
                                    else:
                                        st.markdown("**📱 Phone**")
                                        st.text("Not found")
                                    
                                    if contact_info.get('location'):
                                        st.markdown(f"**📍 Location**")
                                        st.text(contact_info['location'])
                                
                                with contact_cols[1]:
                                    if contact_info.get('linkedin'):
                                        st.markdown(f"**💼 LinkedIn**")
                                        st.markdown(f"[{contact_info['linkedin']}](https://{contact_info['linkedin']})")
                                    
                                    if contact_info.get('github'):
                                        st.markdown(f"**💻 GitHub**")
                                        st.markdown(f"[{contact_info['github']}](https://{contact_info['github']})")
                                    
                                    if contact_info.get('website'):
                                        st.markdown(f"**🌐 Website**")
                                        st.markdown(f"[{contact_info['website']}](https://{contact_info['website']})")
                                
                                if not any([contact_info.get('email'), contact_info.get('phone')]):
                                    st.warning("⚠️ No contact information found in resume. Manual review recommended.")
                                else:
                                    st.success("✅ Contact information extracted successfully!")
                    
                    with col2:
                        if st.button(f"📄 View Resume", key=f"view_{candidate['candidate_name']}_{i}"):
                            with st.container():
                                st.text_area(
                                    "Resume Content",
                                    candidate['text'][:1000] + "...",
                                    height=200,
                                    key=f"resume_text_{candidate['candidate_name']}_{i}"
                                )
                    with col3:
                        if candidate['percentage_score'] >= 50:
                            if st.button(f"⭐ Add to Shortlist", key=f"shortlist_{candidate['candidate_name']}_{i}"):
                                st.success("Added to shortlist!")
                        else:
                            if st.button(f"💾 Save for Later", key=f"save_{candidate['candidate_name']}_{i}"):
                                st.info("Saved for future reference!")
            
            st.markdown("")  # Add spacing between categories


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
    st.title("Candidate Recommendation Engine")
    st.markdown(
        "Upload resumes and enter a job description to find the best candidates using AI-powered matching."
    )
    
    # Load models
    embedding_engine, summarizer = load_models()
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Category filter
        st.markdown("### 🎯 Score Thresholds")
        st.markdown("""
        - **🌟 Perfect Match**: 90-100%
        - **⭐ Ideal Candidate**: 70-90%
        - **✅ Good Candidate**: 50-70%
        - **👍 Okay Candidate**: 20-50%
        - **❌ Not Recommended**: <20%
        """)
        
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
            st.session_state.processed_files = []
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
