import os
import re

import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# 1. Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Tailor & ATS Engine",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 AI Resume Tailor & ATS Optimization Engine")
st.caption(
    "Powered by Google Gemini, Scikit-Learn, and an "
    "Agentic Resume Quality Assurance Workflow"
)


# ---------------------------------------------------------
# 2. Gemini API key configuration
# ---------------------------------------------------------
gemini_api_key = os.environ.get("GEMINI_API_KEY")

if not gemini_api_key:
    try:
        gemini_api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        gemini_api_key = None

if not gemini_api_key:
    st.error(
        "GEMINI_API_KEY is missing. Add it to "
        ".streamlit/secrets.toml or your environment variables."
    )
    st.stop()

# Use a stable Gemini Flash model. If this model is unavailable to
# your account, replace it with a model listed in Google AI Studio.
MODEL_NAME = "gemini-3.5-flash-lite"
client = genai.Client(api_key=gemini_api_key)


def generate_text(prompt: str) -> str:
    """Call Gemini and return generated text, with a clear empty-response error."""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    response_text = getattr(response, "text", None)
    if not response_text or not response_text.strip():
        raise RuntimeError(
            "Gemini returned an empty response. Please retry with a shorter input."
        )
    return response_text.strip()


def parse_review(text: str) -> tuple[str, str]:
    """Parse STATUS and ISSUES fields from a QA response."""
    status_match = re.search(
        r"STATUS:\s*(PASS|REVISE)\b",
        text,
        re.IGNORECASE,
    )
    issues_match = re.search(
        r"ISSUES:\s*(.*)",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    status = status_match.group(1).upper() if status_match else "REVISE"
    issues = issues_match.group(1).strip() if issues_match else text.strip()
    return status, issues


# ---------------------------------------------------------
# 3. Main workspace
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 Upload Your Resume File")
    uploaded_file = st.file_uploader(
        "Choose a PDF resume file (.pdf)",
        type=["pdf"],
    )

    resume_text = ""
    if uploaded_file is not None:
        try:
            pdf_reader = PdfReader(uploaded_file)
            extracted_pages = [
                page.extract_text() or "" for page in pdf_reader.pages
            ]
            resume_text = "\n".join(extracted_pages).strip()

            if resume_text:
                st.success("📄 Resume uploaded and parsed successfully!")
            else:
                st.error(
                    "The PDF contains no extractable text. "
                    "If it is scanned, OCR may be required."
                )
        except Exception as exc:
            st.error(f"Error reading PDF: {exc}")

with col2:
    st.subheader("💼 Target Job Description")
    jd_text = st.text_area(
        "Paste the Job Description:",
        height=180,
        placeholder="Paste the job requirements here...",
    )


# ---------------------------------------------------------
# 4. Analysis and agentic workflow
# ---------------------------------------------------------
if st.button("🔥 Run ATS Analysis & Tailor Resume", type="primary"):
    if not resume_text:
        st.error("Please upload a readable PDF resume.")
        st.stop()

    if not jd_text.strip():
        st.error("Please paste the target Job Description.")
        st.stop()

    try:
        with st.spinner("Running ATS analysis and AI agents..."):

            # STEP 1: Traditional NLP similarity
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
            similarity_matrix = cosine_similarity(
                tfidf_matrix[0:1],
                tfidf_matrix[1:2],
            )
            math_match_percentage = round(
                float(similarity_matrix[0][0]) * 100,
                1,
            )

            # AGENT 1: ATS analyst
            critic_prompt = f"""
You are a strict technical recruiter and resume analyst.

Compare the original resume with the target job description.

ORIGINAL RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

Identify up to four important relevant skills or keywords that appear
to be missing from the original resume.

Rules:
- Never assume the candidate has a skill that is not shown.
- Missing skills are recommendations, not confirmed abilities.
- Do not invent candidate information.
- Give an estimated resume-job-description alignment score from 0 to 100.
- List at most four relevant missing skills or keywords.

Return exactly this format:
SCORE: [integer from 0 to 100]
MISSING SKILLS: [comma-separated list, or NONE]
"""
            critic_text = generate_text(critic_prompt)

            score_match = re.search(
                r"SCORE:\s*(\d{1,3})",
                critic_text,
                re.IGNORECASE,
            )
            skills_match = re.search(
                r"MISSING SKILLS:\s*(.*)",
                critic_text,
                re.IGNORECASE | re.DOTALL,
            )

            llm_ats_score = (
                max(0, min(100, int(score_match.group(1))))
                if score_match
                else 50
            )

            raw_skills = (
                skills_match.group(1).strip()
                if skills_match
                else "NONE"
            )
            missing_skills_list = [
                skill.strip()
                for skill in raw_skills.split(",")
                if skill.strip() and skill.strip().upper() != "NONE"
            ][:4]

            # AGENT 2: Resume writer
            writer_prompt = f"""
You are an expert resume writer.

Tailor the original resume to the target job description.

ORIGINAL RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

Potential skill gaps identified by the ATS analyst:
{', '.join(missing_skills_list) if missing_skills_list else 'None identified'}

Rules:
1. Preserve the candidate's actual experience and qualifications.
2. Never invent skills, certifications, achievements, or metrics.
3. Do not claim the candidate possesses a missing skill unless the
   original resume supports that claim.
4. Improve wording and highlight relevant existing experience.
5. Include relevant keywords naturally when factually supported.
6. Preserve important information from the original resume.
7. Return the complete tailored resume in Markdown.
"""
            tailored_resume_markdown = generate_text(writer_prompt)

            # AGENT 3: Initial resume quality assurance review
            review_prompt = f"""
You are a strict resume quality assurance reviewer.

Compare the original resume, job description, and tailored resume.

ORIGINAL RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

TAILORED RESUME:
{tailored_resume_markdown}

Check for:
1. Unsupported skills, experience, qualifications, or achievements.
2. Invented numerical metrics or certifications.
3. Missing important and relevant job-description keywords.
4. Whether the resume remains relevant to the target role.
5. Whether wording changes the meaning of the original facts.

Rules:
- Do not treat every missing keyword as a factual error.
- Missing skills should be recommended, not falsely claimed.
- Flag unsupported claims and important omissions.
- If the resume is sufficiently accurate and relevant, PASS it.

Return exactly this format:
STATUS: PASS or REVISE
ISSUES: A concise list of problems, or NONE
"""
            review_text = generate_text(review_prompt)
            review_status, review_issues = parse_review(review_text)

            revision_performed = False
            final_review_completed = False

            # Conditional decision: revise once if the reviewer requests it
            if review_status == "REVISE":
                revision_prompt = f"""
You are revising a resume based on quality-review feedback.

ORIGINAL RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

CURRENT TAILORED RESUME:
{tailored_resume_markdown}

REVIEWER FEEDBACK:
{review_issues}

Instructions:
- Return the complete revised resume in Markdown.
- Fix the reviewer's valid concerns.
- Never invent skills, qualifications, experience, achievements, or metrics.
- Do not claim missing skills the candidate does not possess.
- Preserve relevant information from the original resume.
- If an issue cannot be resolved from the available facts, avoid
  making an unsupported claim.
"""
                revised_text = generate_text(revision_prompt)

                if revised_text.strip():
                    tailored_resume_markdown = revised_text
                    revision_performed = True

                # Final QA review of the revised resume
                final_review_prompt = f"""
Recheck the revised resume against the original information.

ORIGINAL RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

REVISED RESUME:
{tailored_resume_markdown}

Check for unsupported claims and important relevant omissions.

Return exactly this format:
STATUS: PASS or REVISE
ISSUES: A concise list of remaining problems, or NONE
"""
                final_review_text = generate_text(final_review_prompt)
                review_status, review_issues = parse_review(final_review_text)
                final_review_completed = True

        # -------------------------------------------------
        # 5. Display results
        # -------------------------------------------------
        st.success("Resume processing completed!")

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Gemini AI ATS Score", f"{llm_ats_score} / 100")
        metric_col2.metric("TF-IDF Similarity", f"{math_match_percentage}%")

        chart_data = pd.DataFrame(
            {
                "Evaluation Metric": [
                    "Gemini AI Score",
                    "TF-IDF Similarity",
                ],
                "Percentage Match": [
                    llm_ats_score,
                    math_match_percentage,
                ],
            }
        )
        fig = px.bar(
            chart_data,
            x="Evaluation Metric",
            y="Percentage Match",
            color="Evaluation Metric",
            text="Percentage Match",
            range_y=[0, 100],
        )
        st.plotly_chart(fig, use_container_width=True)

        # Keyword gaps
        st.subheader("🎯 Identified ATS Keyword Gaps")
        if missing_skills_list:
            gap_cols = st.columns(len(missing_skills_list))
            for index, skill in enumerate(missing_skills_list):
                gap_cols[index].warning(f"Potential gap: {skill}")
        else:
            st.info("No missing skills were identified by the analyst.")

        # QA results
        st.markdown("---")
        st.subheader("🛡️ Resume Quality Assurance Agent")
        if review_status == "PASS":
            st.success("The quality reviewer marked the resume as PASS.")
        else:
            st.warning(
                "The reviewer still identified potential issues. "
                "Inspect the feedback before using the resume."
            )

        if revision_performed:
            st.info("The system revised the resume once based on reviewer feedback.")
        else:
            st.info("The initial tailored resume did not trigger a revision.")

        if final_review_completed:
            st.caption("The revised resume received a final review.")

        st.markdown("**Latest reviewer feedback:**")
        st.write(review_issues)

        # Tailored resume and download
        st.markdown("---")
        st.subheader("✨ Your Tailored, ATS-Optimized Resume")
        st.markdown(tailored_resume_markdown)

        st.download_button(
            label="📥 Download Tailored Resume Text",
            data=tailored_resume_markdown,
            file_name="Tailored_Resume.txt",
            mime="text/plain",
        )

    except Exception as exc:
        st.error(f"An error occurred during processing: {exc}")
        st.info(
            "If this is an API error, verify that your Gemini API key is active, "
            "the model is available to your account, and your API quota has not "
            "been exceeded."
        )
