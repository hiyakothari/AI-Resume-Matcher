import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import openai
from typing import Dict, List, Tuple
import io

# Page configuration
st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .high-score {
        background-color: #d4edda;
        color: #155724;
    }
    .medium-score {
        background-color: #fff3cd;
        color: #856404;
    }
    .low-score {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

class ResumeAnalyzer:
    """Main class for analyzing resume-job description match"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=500
        )
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_keywords(self, text: str, top_n: int = 20) -> List[str]:
        """Extract top keywords using TF-IDF"""
        cleaned = self.preprocess_text(text)
        try:
            tfidf_matrix = self.vectorizer.fit_transform([cleaned])
            feature_names = self.vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            top_indices = scores.argsort()[-top_n:][::-1]
            return [feature_names[i] for i in top_indices if scores[i] > 0]
        except:
            return []
    
    def calculate_match_score(self, resume: str, job_desc: str) -> Tuple[float, Dict]:
        """Calculate similarity score between resume and job description"""
        resume_clean = self.preprocess_text(resume)
        job_clean = self.preprocess_text(job_desc)
        
        # Calculate cosine similarity
        try:
            tfidf_matrix = self.vectorizer.fit_transform([resume_clean, job_clean])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            score = round(similarity * 100, 2)
        except:
            score = 0.0
        
        # Extract keywords from both
        resume_keywords = set(self.extract_keywords(resume, 30))
        job_keywords = set(self.extract_keywords(job_desc, 30))
        
        # Find matching and missing keywords
        matching_keywords = resume_keywords.intersection(job_keywords)
        missing_keywords = job_keywords - resume_keywords
        
        analysis = {
            'score': score,
            'matching_keywords': list(matching_keywords),
            'missing_keywords': list(missing_keywords),
            'resume_keywords': list(resume_keywords),
            'job_keywords': list(job_keywords)
        }
        
        return score, analysis

def get_ai_suggestions(resume: str, job_desc: str, analysis: Dict, api_key: str) -> str:
    """Get AI-powered suggestions using OpenAI API"""
    try:
        openai.api_key = api_key
        
        prompt = f"""You are an expert career coach and resume reviewer. Analyze the following resume against the job description and provide actionable suggestions.

Job Description Keywords: {', '.join(analysis['job_keywords'][:15])}
Resume Keywords: {', '.join(analysis['resume_keywords'][:15])}
Missing Keywords: {', '.join(analysis['missing_keywords'][:10])}
Match Score: {analysis['score']}%

Based on this analysis, provide:
1. 3-5 specific improvements to increase the match score
2. Skills or keywords that should be added (if genuinely possessed)
3. How to better highlight relevant experience
4. Formatting or structure suggestions

Keep suggestions concise and actionable."""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional resume optimization expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Could not generate AI suggestions. Error: {str(e)}\n\nPlease check your API key and try again."

# Main App
def main():
    st.markdown('<div class="main-header">📄 AI Resume Matcher</div>', unsafe_allow_html=True)
    st.markdown("### Compare your resume with job descriptions and get AI-powered suggestions")
    
    # Sidebar for API key
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input("OpenAI API Key ", type="password", 
                                help="Enter your OpenAI API key for AI-powered suggestions")
        st.markdown("---")
        st.markdown("### 📊 How It Works")
        st.markdown("""
        1. **Upload** your resume text
        2. **Paste** the job description
        3. **Analyze** to see your match score
        4. **Get** AI-powered suggestions
        """)
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Use complete resume text
        - Include full job description
        - Higher scores indicate better matches
        - Focus on missing keywords
        """)
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Your Resume")
        resume_text = st.text_area(
            "Paste your resume text here",
            height=400,
            placeholder="Copy and paste your resume content here..."
        )
    
    with col2:
        st.subheader("💼 Job Description")
        job_text = st.text_area(
            "Paste the job description here",
            height=400,
            placeholder="Copy and paste the job description here..."
        )
    
    # Analyze button
    if st.button("🔍 Analyze Match", type="primary", use_container_width=True):
        if not resume_text or not job_text:
            st.error("⚠️ Please provide both resume and job description!")
            return
        
        with st.spinner("Analyzing your resume..."):
            analyzer = ResumeAnalyzer()
            score, analysis = analyzer.calculate_match_score(resume_text, job_text)
            
            # Display score with color coding
            score_class = "high-score" if score >= 70 else "medium-score" if score >= 50 else "low-score"
            st.markdown(f"""
                <div class="score-box {score_class}">
                    Match Score: {score}%
                </div>
            """, unsafe_allow_html=True)
            
            # Display analysis in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "✅ Matching Keywords", "❌ Missing Keywords", "🤖 AI Suggestions"])
            
            with tab1:
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Match Score", f"{score}%")
                with col_b:
                    st.metric("Matching Keywords", len(analysis['matching_keywords']))
                with col_c:
                    st.metric("Missing Keywords", len(analysis['missing_keywords']))
                
                st.markdown("---")
                if score >= 70:
                    st.success("🎉 Excellent match! Your resume aligns well with the job description.")
                elif score >= 50:
                    st.warning("📈 Good match, but there's room for improvement. Check the missing keywords.")
                else:
                    st.error("⚠️ Low match score. Consider revising your resume to better align with the job requirements.")
            
            with tab2:
                st.subheader("✅ Keywords Found in Your Resume")
                if analysis['matching_keywords']:
                    # Display as tags
                    keywords_html = " ".join([f"<span style='background-color:#d4edda; padding:5px 10px; margin:5px; border-radius:5px; display:inline-block;'>{kw}</span>" 
                                            for kw in analysis['matching_keywords'][:20]])
                    st.markdown(keywords_html, unsafe_allow_html=True)
                else:
                    st.info("No matching keywords found.")
            
            with tab3:
                st.subheader("❌ Keywords Missing from Your Resume")
                if analysis['missing_keywords']:
                    keywords_html = " ".join([f"<span style='background-color:#f8d7da; padding:5px 10px; margin:5px; border-radius:5px; display:inline-block;'>{kw}</span>" 
                                            for kw in analysis['missing_keywords'][:20]])
                    st.markdown(keywords_html, unsafe_allow_html=True)
                    st.info("💡 Consider adding these keywords to your resume if they match your actual skills and experience.")
                else:
                    st.success("Great! Your resume covers all major keywords from the job description.")
            
            with tab4:
                st.subheader("🤖 AI-Powered Suggestions")
                if api_key:
                    with st.spinner("Generating personalized suggestions..."):
                        suggestions = get_ai_suggestions(resume_text, job_text, analysis, api_key)
                        st.markdown(suggestions)
                else:
                    st.info("🔑 Enter your OpenAI API key in the sidebar to get AI-powered suggestions!")
                    st.markdown("""
                    **Manual Suggestions Based on Analysis:**
                    
                    1. **Add Missing Keywords**: Review the missing keywords tab and incorporate relevant ones into your resume
                    2. **Quantify Achievements**: Use numbers and metrics to demonstrate impact
                    3. **Tailor Your Summary**: Align your professional summary with the job requirements
                    4. **Highlight Relevant Skills**: Emphasize skills that match the job description
                    5. **Use Action Verbs**: Start bullet points with strong action verbs
                    """)

if __name__ == "__main__":
    main()
