
# AI-Assisted Data Cleaning Tool (Streamlit)

A client-ready Streamlit app that cleans and preprocesses tabular data (CSV) with both **rule-based** steps and **AI suggestions**.
The AI assistant (via OpenAI) recommends column types, imputations, encodings, and transformations based on your dataset.

## Features
- Duplicate removal
- Missing value handling (numeric & categorical)
- Text standardization (trim, case, punctuation cleanup)
- Date parsing & type fixing
- Outlier detection/removal (IQR)
- Encoding (One-Hot) & scaling (StandardScaler) options
- AI suggestions for schema & steps (uses `OPENAI_API_KEY` env var)
- Download cleaned CSV + JSON report

## Quickstart

```bash
# 1) Create virtual env (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Set your OpenAI key (optional but recommended)
export OPENAI_API_KEY=sk-...   # macOS/Linux
# or on Windows PowerShell:
# setx OPENAI_API_KEY "sk-..."

# 4) Run the app
streamlit run app.py
```

Open the URL shown in terminal (usually http://localhost:8501), upload a CSV, preview suggestions, and download the cleaned data.

## Project Structure
```
ai_data_cleaner/
├─ app.py            # Streamlit UI
├─ cleaner.py        # Rule-based cleaning pipeline
├─ ai_helper.py      # LLM-powered suggestions
├─ utils.py          # Helpers (reports, types, etc.)
├─ requirements.txt  # Dependencies
├─ sample_data.csv   # Example dataset
└─ README.md
```

## Notes
- The app works **without** the OpenAI key; AI suggestions will simply be skipped.
- For very large files, consider sampling first or using chunked reading.
"# CleanMind-AI-Intelligent-Data-Preprocessing-Tool" 
