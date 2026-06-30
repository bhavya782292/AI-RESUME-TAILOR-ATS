import streamlit as st
import pandas as pd
import plotly.express as px
from mistralai.client import Mistral
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
from pypdf import PdfReader  # Library to handle PDF text extraction

# 1. Page Configuration
st.set_page_config(page_title="AI Resume Tailor & ATS Engine", layout="wide")
st.title("🚀 AI Resume Tailor & ATS Optimization Engine")
st.caption("Powered by Mistral AI, Scikit-Learn, and Pandas Data Visualizations")

# 2. Look directly at your terminal variable (Hidden from UI)
mistral_api_key = os.environ.get("MISTRAL_API_KEY")

if not mistral_api_key:
    st.error("⚠️ Setup Error: MISTRAL_API_KEY is not loaded in your terminal yet.")
    st.stop()

# 3. Main Workspace Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 Upload Your Resume File")
    # Updated to accept PDF files natively
    uploaded_file = st.file_uploader("Choose a PDF resume file (.pdf)", type=["pdf"])
    
    resume_text = ""
    if uploaded_file is not None:
        try:
            # Read and extract text from all pages of the PDF
            pdf_reader = PdfReader(uploaded_file)
            extracted_pages = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
            resume_text = "\n".join(extracted_pages)
            
            if resume_text.strip():
                st.success("📄 PDF Resume successfully uploaded and parsed!")
            else:
                st.error("⚠️ The uploaded PDF seems to be blank or contains only images.")
        except Exception as e:
            st.error(f"Error reading PDF file: {str(e)}")

with col2:
    st.subheader("💼 Paste Target Job Description")
    jd_text = st.text_area("Job Description:", height=180, placeholder="Paste job requirements here...")

# 4. Processing Core Engine
if st.button("🔥 Run ATS Analysis & Tailor Resume"):
    if not resume_text:
        st.error("Please upload your PDF resume file to proceed.")
    elif not jd_text:
        st.error("Please paste the target job description to proceed.")
    else:
        with st.spinner("Executing Pipeline..."):
            try:
                # Math Vector Engine
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
                similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
                math_match_percentage = round(float(similarity_matrix[0][0]) * 100, 1)

                # Mistral AI Agents
                client = Mistral(api_key=mistral_api_key)
                
                critic_prompt = f"""
                You are a strict technical recruiter and ATS software. 
                Compare the following Resume against the Job Description.
                Resume: {resume_text}
                Job Description: {jd_text}
                Provide your response in EXACTLY this format:
                SCORE: [Provide an integer out of 100]
                MISSING SKILLS: [Provide a comma-separated list of the top 4 missing tools]
                """
                
                critic_response = client.chat.complete(
                    model="mistral-large-latest",
                    messages=[{"role": "user", "content": critic_prompt}]
                )
                critic_text = critic_response.choices.message.content

                score_match = re.search(r"SCORE:\s*(\d+)", critic_text)
                skills_match = re.search(r"MISSING SKILLS:\s*(.*)", critic_text)
                
                llm_ats_score = int(score_match.group(1)) if score_match else 50
                missing_skills_list = skills_match.group(1).split(",") if skills_match else ["Docker", "AWS"]

                writer_prompt = f"""
                You are an expert resume writer. Tailor this Resume to match this Job Description.
                Inject these missing keywords: {', '.join(missing_skills_list)} smoothly.
                Original Resume: {resume_text}
                Job Description: {jd_text}
                """
                
                writer_response = client.chat.complete(
                    model="mistral-large-latest",
                    messages=[{"role": "user", "content": writer_prompt}]
                )
                tailored_resume_markdown = writer_response.choices.message.content

                # UI Display
                st.success("Analysis Complete!")
                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric("Mistral AI Score", f"{llm_ats_score} / 100")
                metric_col2.metric("Scikit-Learn Match", f"{math_match_percentage}%")
                
                chart_data = pd.DataFrame({
                    "Evaluation Metric": ["Mistral AI Score", "Scikit-Learn Vector Match"],
                    "Percentage Match": [llm_ats_score, math_match_percentage]
                })
                fig = px.bar(chart_data, x="Evaluation Metric", y="Percentage Match", color="Evaluation Metric", text="Percentage Match", range_y=[0, 100])
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("🎯 Identified ATS Keyword Gaps")
                gap_cols = st.columns(len(missing_skills_list))
                for index, skill in enumerate(missing_skills_list):
                    gap_cols[index % len(gap_cols)].error(f"❌ Missing: {skill.strip()}")

                st.markdown("---")
                st.subheader("✨ Your Tailored, ATS-Optimized Resume Text")
                st.markdown(tailored_resume_markdown)
                
                st.download_button(
                    label="📥 Download Tailored Resume Text",
                    data=tailored_resume_markdown,
                    file_name="Tailored_Resume.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
