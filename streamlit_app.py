import streamlit as st
# Set page config at the very beginning
st.set_page_config(page_title="Resume Ranker Chatbot", page_icon="📄")

import spacy
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd
import os
import logging
from typing import Tuple, List, Optional
import tempfile
import matplotlib
matplotlib.use('Agg')  # Add this line right after importing matplotlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize spaCy model
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}")
        st.error("Failed to load language model. Please check if it's installed correctly.")
        return None

nlp = load_nlp_model()

def extract_text_from_pdf(pdf_file) -> Optional[str]:
    """Extract text from uploaded PDF file with error handling"""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(pdf_file.getbuffer())
            with open(tmp_file.name, "rb") as pdf:
                pdf_reader = PyPDF2.PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
        os.unlink(tmp_file.name)
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return None

def extract_entities(text: str) -> Tuple[List[str], List[str]]:
    """
    Extract emails and names from text using regex patterns
    Args:
        text (str): Input text from resume
    Returns:
        Tuple[List[str], List[str]]: Lists of extracted emails and names
    """
    try:
        # Extract emails using more precise regular expression
        emails = re.findall(r'\S+@\S+', text)
        
        # Extract names using a simple pattern (assuming "First Last" format)
        # Look for capitalized words at the start of lines or after spaces
        names = re.findall(r'^([A-Z][a-z]+)\s+([A-Z][a-z]+)', text, re.MULTILINE)
        
        # Process names into a single list of full names
        formatted_names = []
        if names:
            formatted_names = [" ".join(name_tuple) for name_tuple in names]
            
            # Remove duplicates while preserving order
            formatted_names = list(dict.fromkeys(formatted_names))
            
            # Log found names for debugging
            logger.info(f"Found names: {formatted_names}")
        
        # Clean and validate emails
        clean_emails = [
            email.strip()
            for email in emails
            if '@' in email and '.' in email.split('@')[1]
        ]
        
        return clean_emails, formatted_names
    except Exception as e:
        logger.error(f"Error in extract_entities: {e}")
        return [], []

def rank_resumes(job_description: str, resumes_data: List[Tuple]) -> pd.DataFrame:
    """Rank resumes using TF-IDF and cosine similarity"""
    try:
        tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        job_desc_vector = tfidf_vectorizer.fit_transform([job_description])
        
        ranked_results = []
        for resume_text, names, emails in resumes_data:
            resume_vector = tfidf_vectorizer.transform([resume_text])
            similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0] * 100
            
            name = names[0] if names else "N/A"
            email = emails[0] if emails else "N/A"
            
            ranked_results.append({
                "Name": name,
                "Email": email,
                "Match Score": similarity  # Store as float, not string
            })
        
        return pd.DataFrame(ranked_results)
    except Exception as e:
        logger.error(f"Error ranking resumes: {e}")
        return pd.DataFrame()

def main():
    st.title("AI Resume Ranker")
    
    # Updated markdown with precise core features
    st.markdown("""
    Welcome Good Day! This tool offers:
                
    • Smart Resume Analysis - Uses NLP to evaluate resumes against job requirements
                
    • Automated Contact Extraction - Identifies candidate names and email addresses
                
    • Percentage-Based Matching - Ranks candidates with similarity scores and downloadable results
    """)
    
    # Job Description Input
    job_description = st.text_area(
        "Please paste your job description here:",
        height=150,
        key="job_desc"
    )
    
    # File Upload
    uploaded_files = st.file_uploader(
        "Upload resumes (PDF format only)",
        type="pdf",
        accept_multiple_files=True
    )
    
    if st.button("Analyze Resumes"):
        if not job_description:
            st.warning("Please provide a job description.")
            return
            
        if not uploaded_files:
            st.warning("Please upload at least one resume.")
            return
            
        with st.spinner("Processing resumes... Please wait."):
            resumes_data = []
            
            # Process each resume
            for uploaded_file in uploaded_files:
                text = extract_text_from_pdf(uploaded_file)
                if text:
                    emails, names = extract_entities(text)
                    resumes_data.append((text, names, emails))
                else:
                    st.error(f"Failed to process {uploaded_file.name}")
            
            if resumes_data:
                # Rank resumes
                results_df = rank_resumes(job_description, resumes_data)
                
                if not results_df.empty:
                    st.success("Analysis complete! Here are the results:")
                    
                    try:
                        # Try to display with gradient
                        st.dataframe(
                            results_df.style
                            .background_gradient(
                                subset=['Match Score'],
                                cmap='RdYlGn'
                            )
                            .format({'Match Score': '{:.2f}%'})
                        )
                    except Exception as e:
                        logger.warning(f"Gradient display failed: {e}")
                        # Fallback to simple display
                        formatted_df = results_df.copy()
                        formatted_df['Match Score'] = formatted_df['Match Score'].apply(lambda x: f"{x:.2f}%")
                        st.dataframe(formatted_df)
                    
                    # Download results
                    formatted_df = results_df.copy()
                    formatted_df['Match Score'] = formatted_df['Match Score'].apply(lambda x: f"{x:.2f}%")
                    csv = formatted_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="resume_rankings.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Failed to rank resumes. Please try again.")

if __name__ == "__main__":
    main()
