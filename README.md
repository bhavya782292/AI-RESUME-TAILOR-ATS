# AI Resume Tailor & ATS Optimization Engine

**Live App:** https://imydnjw8qsjfbvegiwzah7.streamlit.app/  
**GitHub Repository:** https://github.com/bhavya782292/AI-RESUME-TAILOR-ATS

An AI-powered Streamlit application that helps tailor a resume to a job description. It combines resume text extraction, keyword-based similarity analysis, Gemini-powered feedback, and a review-and-revision workflow to help users identify gaps and improve resume alignment.

## Features

- **Resume upload and text extraction:** Upload a resume PDF and extract its text for analysis.
- **Job-description matching:** Compare resume content with a job description using TF-IDF and cosine similarity.
- **Gemini-powered analysis:** Generate ATS-oriented feedback and suggestions using the Gemini API.
- **Resume tailoring:** Produce an improved, job-description-aligned version while avoiding unsupported claims.
- **Quality review:** Review the generated resume for issues and use feedback in a limited revision step.
- **Streamlit interface:** Use the app through the live web interface.

> Results are estimates and suggestions, not a guarantee of passing an ATS or getting an interview. Always review generated content for accuracy before using it.

## Tech Stack

- Python
- Streamlit
- Google Gemini API (`google-genai`)
- PyPDF
- scikit-learn (TF-IDF and cosine similarity)
- pandas
- Plotly

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/bhavya782292/AI-RESUME-TAILOR-ATS.git
cd AI-RESUME-TAILOR-ATS
```

### 2. Create and activate a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create `.streamlit/secrets.toml` locally:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Alternatively, configure `GEMINI_API_KEY` in your deployment platform's secrets settings.

**Security:** Never commit `.streamlit/secrets.toml`, API keys, or other credentials to GitHub. Keep secrets excluded through `.gitignore`. If a key is exposed, revoke it and generate a replacement.

### 5. Start the app

```bash
streamlit run app.py
```

## Deployment

The app is deployed on Streamlit Community Cloud:

**Open the live app:** https://imydnjw8qsjfbvegiwzah7.streamlit.app/

The deployed app uses the GitHub repository's configured branch and entry point. After pushing changes to the connected branch, Streamlit Cloud normally rebuilds the app automatically. Check the app's deployment logs if a build or runtime error occurs.

## Limitations

- ATS scores and similarity scores are approximate indicators, not official ATS results.
- Output quality depends on the supplied resume, job description, model availability, and API limits.
- Review all generated claims, dates, skills, and experience before submitting a resume.
- Gemini API errors such as temporary `503 UNAVAILABLE` responses can occur; retry later if the service is temporarily unavailable.

## Project Structure

```text
AI-RESUME-TAILOR-ATS/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml  # local only; do not commit
```

## Author

**Bhavya Kukreja**

- GitHub: https://github.com/bhavya782292
