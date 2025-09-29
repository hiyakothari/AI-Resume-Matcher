# AI Resume Matcher

An intelligent tool that helps job seekers optimize their resumes by comparing them with job descriptions using Natural Language Processing (NLP) and AI. Get instant feedback on how well your resume matches a job posting, along with actionable suggestions to improve your chances.


## 🌟 Features

Smart Matching Algorithm: Uses TF-IDF vectorization and cosine similarity to calculate match scores
Keyword Analysis: Automatically extracts and compares keywords from both resume and job description
Missing Keywords Detection: Identifies important keywords you should consider adding
AI-Powered Suggestions: Get personalized recommendations using OpenAI's GPT models
Visual Feedback: Color-coded scores and intuitive UI for easy understanding
Real-time Analysis: Instant results as you input your text


## 📖 Usage

Paste Your Resume: Copy your resume text into the left text area
Paste Job Description: Copy the target job description into the right text area
Click Analyze: Hit the "Analyze Match" button
Review Results:

Check your match score
Review matching keywords
Identify missing keywords
Get AI-powered suggestions (requires API key)


## 📊 Understanding Your Score

70-100%: Excellent match! Your resume aligns well
50-69%: Good match, but room for improvement
Below 50%: Consider significant revisions



## Getting OpenAI API Key 

Visit OpenAI Platform
Sign up or log in
Navigate to API keys section
Create new secret key
Enter the key in the app's sidebar

Note: Basic matching functionality works without an API key!


## 🛠️ Technology Stack

Frontend: Streamlit
NLP/ML: Scikit-learn (TF-IDF, Cosine Similarity)
AI Integration: OpenAI API (GPT-3.5-turbo)
Data Processing: NumPy, Regex
Language: Python 3.8+


## How It Works

Text Preprocessing: Cleans and normalizes input text
Feature Extraction: Uses TF-IDF to convert text to numerical vectors
Similarity Calculation: Computes cosine similarity between resume and job description vectors
Keyword Extraction: Identifies top keywords using TF-IDF scores
Gap Analysis: Finds keywords present in job description but missing from resume
AI Enhancement: Sends analysis to GPT for personalized suggestions

