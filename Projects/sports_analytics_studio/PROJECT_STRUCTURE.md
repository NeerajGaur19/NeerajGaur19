# Project structure

```text
sports_analytics_studio/
├── app.py                      # Streamlit app, home navigation, task pages
├── requirements.txt            # Python dependencies
├── .env.example                # Optional Gemini configuration template
├── .gitignore
├── README.md
├── PROJECT_STRUCTURE.md
├── src/
│   ├── __init__.py
│   ├── data_utils.py            # CSV/Excel loading, schema info, feature engineering
│   ├── ml_utils.py              # Model pipelines, metrics, clustering, anomaly detection
│   ├── ai_utils.py              # LangChain/Gemini and ChromaDB retrieval
│   └── generate_samples.py      # Rebuild synthetic sample datasets
└── data/
    └── sample/                  # Synthetic demo CSVs
```

## Key design choices

- The active dataframe and original dataframe are stored separately in Streamlit session state.
- Every supported data-changing action pushes a snapshot onto a bounded undo stack.
- Reset restores the initially loaded dataset; loading a new dataset establishes a new baseline.
- ML preprocessing (imputation, scaling, one-hot encoding) occurs inside sklearn pipelines to reduce leakage from preprocessing.
- ChromaDB retrieves explanatory sports analytics notes; Pandas handles structured tabular data.
- Gemini is optional. The app remains usable without an API key.
