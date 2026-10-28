# Sports Analytics Studio (SAS)

A GitHub-ready Streamlit application for sports data exploration, ML experiments, and Gemini-assisted analysis.

## Features
- Home dashboard with links to five task pages plus a Data Laboratory.
- Upload CSV / Excel datasets, preview data, inspect shape/types/head/tail, and export cleaned data.
- Null and duplicate-row/column inspection with reversible actions.
- EDA: summary statistics, missingness, distributions, category counts, correlations, scatter plots, box plots, time series.
- Feature engineering suggestions and optional application of numeric derived features.
- Run multiple classification/regression models and compare metrics.
- Injury-risk classification, athlete talent identification, event analytics, anomaly detection.
- Optional LangChain + Gemini natural-language analysis.
- ChromaDB semantic retrieval using LangChain HuggingFaceEmbeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Generated synthetic sample datasets (clearly labeled synthetic).

## Quick start (VS Code)

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env  # Windows; on macOS/Linux use: cp .env.example .env
streamlit run app.py
```

Open the local URL printed by Streamlit.

## Gemini setup (optional)
1. Install dependencies from `requirements.txt`; the first Hugging Face embedding call downloads `sentence-transformers/all-MiniLM-L6-v2` and may take a little time.
2. Create a Gemini API key in Google AI Studio.
3. Add it to `.env` as `GOOGLE_API_KEY=...`.
4. Keep `.env` out of Git; `.gitignore` already excludes it.
5. In the app, open **AI Assistant** and ask a question about the active dataset.

The core app works without an API key. AI answers are advisory and may be unavailable if the key, quota, model, or package configuration is not valid.

## Sample data
On first run, sample CSVs are generated under `data/sample/` if absent:
- `athlete_performance.csv`
- `injury_risk.csv`
- `talent_identification.csv`
- `event_analytics.csv`
- `sensor_anomalies.csv`

These are synthetic demonstration data, not real athlete records and not medically validated.

## GitHub
```bash
git init
git add .
git commit -m "Initial Sports Analytics Studio"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/sports-analytics-studio.git
git push -u origin main
```

Do not commit `.env`, real athlete personal data, or sensitive health records.

## Deployment
For Streamlit Community Cloud, connect the GitHub repository and select `app.py`.
Set `GOOGLE_API_KEY` in the app's Secrets only if using Gemini. Check current dependency compatibility and hosting resource limits before production deployment.

## Important limitations
- The built-in datasets are synthetic and should only be used for UI/model demonstrations.
- Injury-risk outputs are experimental statistical predictions, not diagnoses or return-to-play advice.
- Model metrics are meaningful only when labels, splits, and data quality are appropriate. Small, leaked, imbalanced, or biased datasets can produce misleading scores.
- Avoid random train/test splits for time-dependent predictions; use chronological splits where appropriate.
- ChromaDB is used for retrieval over explanatory notes, not as a replacement for structured analytics.
