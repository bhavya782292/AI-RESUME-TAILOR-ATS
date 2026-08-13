# 🚀 AI Resume Tailor & ATS Optimization Engine

A two-stage LLM-powered resume optimization and ATS evaluation application built with Python, Streamlit, Mistral AI, Scikit-learn, Pandas, Plotly, and PyPDF.

The application analyzes a candidate's resume against a target job description using both deterministic TF-IDF cosine similarity and contextual LLM evaluation. It identifies missing skills, generates ATS-oriented recommendations, and produces a tailored resume based on the identified requirements.

---

## 📌 Overview

Applicant Tracking Systems (ATS) and recruiter screening processes often evaluate resumes based on their alignment with job descriptions.

This project combines two complementary approaches:

1. **Deterministic Text Similarity**  
   Uses TF-IDF vectorization and cosine similarity to calculate the similarity between the resume and target job description.

2. **Contextual LLM Evaluation**  
   Uses Mistral AI to analyze the resume and job description, identify missing skills, and generate a tailored resume.

The result is an interactive Streamlit application that provides resume analysis, skill-gap identification, evaluation metrics, and AI-generated resume tailoring.

---

# ⚙️ System Workflow

```text
                 Resume PDF
                     │
                     ▼
              ┌─────────────┐
              │   PyPDF     │
              │ Text Extract│
              └──────┬──────┘
                     │
                     ▼
                Resume Text
                     │
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────────┐    ┌─────────────────────┐
│ TF-IDF Vectorizer │    │   Mistral AI Stage 1│
└─────────┬─────────┘    │ Recruiter / Critic   │
          │              └──────────┬──────────┘
          ▼                         │
┌───────────────────┐               ▼
│ Cosine Similarity │        Missing Skills /
└─────────┬─────────┘        Capability Gaps
          │                         │
          │                         ▼
          │                ┌─────────────────────┐
          │                │ Mistral AI Stage 2  │
          │                │ Resume Writer       │
          │                └──────────┬──────────┘
          │                           │
          └──────────────┬────────────┘
                         ▼
              ┌──────────────────────┐
              │ Evaluation Dashboard │
              │ + Tailored Resume    │
              └──────────────────────┘
