import streamlit as st
import PyPDF2
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
@st.cache_data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

download_nltk_data()

# Page configuration
st.set_page_config(
    page_title="Resume Ranking System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file using PyPDF2"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip() if text.strip() else "No text found"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def preprocess_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, punctuation, and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Join tokens back into string
    return ' '.join(tokens)

def calculate_scores(job_description, resume_texts):
    """Calculate similarity scores between job description and resumes"""
    # Preprocess texts
    cleaned_job_desc = preprocess_text(job_description)
    cleaned_resumes = [preprocess_text(text) for text in resume_texts]
    
    # Vectorization using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    
    # Fit and transform job description and resumes
    all_texts = [cleaned_job_desc] + cleaned_resumes
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Calculate cosine similarity between job description (index 0) and resumes (indices 1+)
    job_desc_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]
    
    similarities = cosine_similarity(job_desc_vector, resume_vectors).flatten()
    
    # Convert to percentages
    scores = (similarities * 100).round(2)
    
    return scores.tolist()

# Header Section
st.markdown('<h1 class="main-header">📄 Resume Ranking System</h1>', unsafe_allow_html=True)
st.markdown("""
    <p class="sub-header">
    Upload resumes and paste a job description to automatically rank candidates based on their match score. 
    Powered by TF-IDF vectorization and cosine similarity.
    </p>
""", unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📝 Job Description")
    job_description = st.text_area(
        "Paste the job description here:",
        height=200,
        placeholder="Enter job description..."
    )

with col2:
    st.subheader("📁 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Choose PDF resume files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload multiple PDF resumes (max 10 files recommended)"
    )

# Action button
if st.button("🚀 Rank Resumes", type="primary", use_container_width=True):
    if not job_description.strip():
        st.error("⚠️ Please enter a job description first!")
    elif not uploaded_files:
        st.error("⚠️ Please upload at least one PDF resume!")
    else:
        with st.spinner("🔄 Processing resumes and calculating match scores..."):
            try:
                # Extract text from all PDFs
                resume_data = []
                valid_resumes = 0
                
                for uploaded_file in uploaded_files:
                    filename = uploaded_file.name
                    pdf_content = io.BytesIO(uploaded_file.read())
                    
                    text_content = extract_text_from_pdf(pdf_content)
                    resume_data.append({
                        'filename': filename,
                        'text': text_content
                    })
                    
                    if "Error" not in text_content and "No text found" not in text_content:
                        valid_resumes += 1
                
                if valid_resumes == 0:
                    st.error("❌ No valid text could be extracted from any of the uploaded PDFs!")
                    st.info("💡 Tips: Ensure PDFs are text-based (not scanned images) and not password protected.")
                else:
                    # Filter valid resumes and calculate scores
                    valid_resume_data = [item for item in resume_data if "Error" not in item['text']]
                    resume_texts = [item['text'] for item in valid_resume_data]
                    
                    scores = calculate_scores(job_description, resume_texts)
                    
                    # Create results DataFrame
                    results_df = pd.DataFrame({
                        'Rank': range(1, len(scores) + 1),
                        'Candidate': [item['filename'] for item in valid_resume_data],
                        'Match Score (%)': scores
                    })
                    
                    # Sort by score (descending)
                    results_df = results_df.sort_values('Match Score (%)', ascending=False).reset_index(drop=True)
                    results_df['Rank'] = range(1, len(results_df) + 1)
                    
                    # Display metrics
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Resumes", len(uploaded_files))
                    with col2:
                        st.metric("Valid Resumes", valid_resumes)
                    with col3:
                        st.metric("Top Score", f"{results_df['Match Score (%)'].max():.1f}%")
                    
                    # Results table with progress bars
                    st.markdown("## 🏆 Ranking Results")
                    
                    for idx, row in results_df.iterrows():
                        score_color = "🟢" if row['Match Score (%)'] >= 70 else "🟡" if row['Match Score (%)'] >= 50 else "🔴"
                        
                        col1, col2, col3 = st.columns([0.2, 2, 2])
                        with col1:
                            st.markdown(f"**#{row['Rank']}**")
                        with col2:
                            st.markdown(f"**{row['Candidate']}**")
                        with col3:
                            st.progress(row['Match Score (%)'] / 100)
                            st.markdown(f"{score_color} **{row['Match Score (%)']:.1f}%**")
                        
                        st.markdown("---")
                    
                    # Download results as CSV
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"resume_ranking_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Show top 3 recommendations
                    if len(results_df) >= 3:
                        st.markdown("### 🎯 Top 3 Recommendations")
                        top_3 = results_df.head(3)
                        for _, row in top_3.iterrows():
                            st.success(f"#{row['Rank']} - {row['Candidate']} ({row['Match Score (%)']:.1f}%)")
                    
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("Please check your files and try again.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Built with ❤️ using Streamlit, Scikit-learn & PyPDF2<br>
        <small>Upload text-based PDFs for best results</small>
    </div>
    """, 
    unsafe_allow_html=True
)
