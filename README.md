# 🚀 AI Resume Tailor & ATS Optimization Engine

An AI-powered resume optimization and ATS evaluation application built using **Python, Streamlit, Mistral AI, Scikit-learn, Pandas, Plotly, and PyPDF**.

The application analyzes a candidate's resume against a target Job Description (JD), calculates resume-JD similarity, identifies missing skills and keywords, and uses a two-stage LLM workflow to generate a tailored resume.

## 🔗 Links

- 🌐 **Live Demo:** [Try the AI Resume Tailor](https://imydnjw8qsjfbvegiwzah7.streamlit.app/)
- 💻 **GitHub Repository:** [View Source Code](https://github.com/bhavya782292/AI-RESUME-TAILOR-ATS)

---

## 🎯 Project Overview

Many resumes fail to perform well in ATS screening because they do not sufficiently match the skills, keywords, and requirements mentioned in a job description.

This project provides an automated resume analysis and tailoring workflow that:

- 📄 Extracts text from uploaded PDF resumes
- 💼 Accepts a target Job Description
- 🔍 Calculates resume-JD similarity using TF-IDF and Cosine Similarity
- 🤖 Uses Mistral AI for resume evaluation
- 🧩 Identifies missing skills and keywords
- ✍️ Generates a tailored resume using an LLM
- 📊 Displays ATS and similarity metrics
- 📥 Provides downloadable tailored resume content

The system combines **traditional NLP techniques with LLM-based analysis and generation**.

---

## ✨ Key Features

### 📄 PDF Resume Parsing

Users can upload their resume in PDF format.

The application uses **PyPDF** to extract text from the uploaded resume before performing further analysis.

### 💼 Job Description Analysis

Users can paste the target Job Description into the application.

The resume content is then analyzed against the requirements of the provided JD.

### 📊 TF-IDF + Cosine Similarity

The application uses Scikit-learn's:

- `TfidfVectorizer`
- `cosine_similarity`

to calculate the textual similarity between the resume and the target job description.

The resulting similarity score is displayed as a percentage.

### 🤖 Mistral AI Resume Analysis

The application integrates **Mistral AI** to evaluate the resume against the target job description.

The first LLM stage analyzes the resume and identifies:

- Resume-JD alignment
- Missing skills
- Relevant keywords
- ATS-oriented score

### ✍️ AI Resume Tailoring

A second Mistral AI stage uses the analysis from the first stage to generate a tailored version of the resume.

The workflow uses the identified missing skills and keywords to improve alignment with the target JD while maintaining the candidate's existing information.

### 📈 Interactive Streamlit Dashboard

The application provides an interactive interface displaying:

- Mistral AI score
- Scikit-learn similarity score
- Missing skills
- Evaluation visualization
- Tailored resume output

### 📥 Downloadable Output

The generated tailored resume content can be downloaded directly from the application.

---

## 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │   Upload Resume PDF  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       PyPDF          │
                         │   Text Extraction    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Resume Text       │
                         └──────────┬───────────┘
                                    │
                                    │
                 ┌──────────────────┴──────────────────┐
                 │                                     │
                 ▼                                     ▼
       ┌───────────────────┐               ┌─────────────────────┐
       │ Target Job        │               │ TF-IDF Vectorizer   │
       │ Description       │               │ + Cosine Similarity │
       └─────────┬─────────┘               └──────────┬──────────┘
                 │                                    │
                 │                                    ▼
                 │                           Resume-JD Similarity
                 │
                 ▼
       ┌────────────────────────────┐
       │ Mistral AI - Stage 1       │
       │ Recruiter / Critic Analysis│
       └────────────┬───────────────┘
                    │
                    ▼
            ┌──────────────────┐
            │ Missing Skills   │
            │ Keywords + Score │
            └────────┬─────────┘
                     │
                     ▼
       ┌────────────────────────────┐
       │ Mistral AI - Stage 2       │
       │ Resume Writer              │
       └────────────┬───────────────┘
                    │
                    ▼
       ┌────────────────────────────┐
       │   Tailored Resume Output   │
       └────────────┬───────────────┘
                    │
                    ▼
       ┌────────────────────────────┐
       │    Streamlit Dashboard     │
       │ Metrics + Visualization    │
       └────────────────────────────┘
