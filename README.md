# Multi-Agent Resume Optimization & ATS Evaluation Engine

[![Streamlit App](https://streamlit.io)](https://streamlit.app)

A dual-engine automated workspace designed to map engineering profiles against dense enterprise job requirements. The application features a decoupled text evaluation pipeline that evaluates semantic alignment using context-aware LLMs while concurrently verifying absolute keyword densities via traditional text vectorization.

## ⚙️ Core Architecture & Pipeline Execution

Unlike vanilla application wrappers, this tool uses a hybrid evaluation structure to bypass strict applicant screening filters:

1. **Deterministic Vector Similarity Line**: Implements Scikit-Learn's `TfidfVectorizer` to extract raw spatial token metrics from the input texts. It then calculates the exact cosine similarity matrix, giving a direct math metric of keyword match density.
2. **Contextual LLM Evaluation Line**: Utilizes a dual-agent configuration via the `mistral-large-latest` client interface. 
   - **Agent 1 (The Recruiter)** acts as a parser, mapping structural capability gaps and cataloging specific tool deficiencies.
   - **Agent 2 (The Copywriter)** processes the identified feature sets to output an optimized, completely tailored markdown resume that seamlessly injects the missing parameters without inventing false historical data.
3. **Data Profiling Layer**: Aggregates statistical metrics into clean Pandas structured arrays, driving interactive, low-latency UI charts built with Plotly.

---

## 🛠️ Local Environment Workspace Setup

Ensure Python 3.10+ is actively running on your local device.

### 1. Dependencies Installation
Install the necessary package bundle using your default environment utility. For high-speed virtual environments (`uv`), execute:
```bash
uv pip install --system streamlit mistralai scikit-learn pandas plotly pypdf
```
*(Alternatively, utilize standard package management: `pip install -r requirements.txt`)*

### 2. Secure Local Credential Handling
For local execution, configure the target API credentials safely inside your local environment shell to avoid hardcoding exposure.

In your macOS terminal window, export your authorization string:
```bash
export MISTRAL_API_KEY="your_secret_mistral_api_key_here"
```

### 3. Executing the Local Server
Boot up the local Streamlit application module:
```bash
streamlit run app.py
```
The application interface will automatically launch on your local host address at `http://localhost:8501`.

---

## 📊 Feature Breakdown & UI Components
- **Native Document Extractor**: Integrated text layer processing via `pypdf` to parse text strings from standard PDF inputs.
- **Hidden Credentials Ingestion**: Relies entirely on system environment contexts (`os.environ`), leaving zero trace of security keys on the frontend layout during product demos or recording deployment loops.
- **Analytics Visualization Suite**: Real-time rendering comparing conceptual machine matching against rigid keyword math rules to clearly diagnose filter vulnerabilities.
