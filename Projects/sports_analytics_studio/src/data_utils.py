from __future__ import annotations
import io
import numpy as np
import pandas as pd

def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Supported formats are CSV, XLSX, and XLS.")

def dataframe_info(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": df.columns.astype(str),
        "dtype": [str(x) for x in df.dtypes],
        "non_null_count": [int(df[c].notna().sum()) for c in df.columns],
        "null_count": [int(df[c].isna().sum()) for c in df.columns],
        "null_percent": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
        "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })

def duplicate_column_names(df: pd.DataFrame) -> list:
    return df.columns[df.columns.duplicated()].astype(str).tolist()

def duplicate_content_columns(df: pd.DataFrame) -> list:
    # Keep the first column where two columns have exactly identical values.
    seen = {}
    duplicates = []
    for i, col in enumerate(df.columns):
        series = df.iloc[:, i]
        # Hashing handles most common data types; compare exact values after hash match.
        h = pd.util.hash_pandas_object(series, index=False).sum()
        key = (str(series.dtype), int(h), len(series))
        if key in seen and series.equals(df.iloc[:, seen[key]]):
            duplicates.append(i)
        else:
            seen[key] = i
    return duplicates

def apply_basic_feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    created = []
    lower_to_col = {str(c).lower(): c for c in out.columns}
    # Only add features when the source columns exist; do not silently overwrite.
    pairs = [
        (("distance_km", "duration_min"), "avg_speed_kmh",
         lambda d, t: d / (t / 60).replace(0, np.nan)),
        (("total_distance_m", "duration_seconds"), "avg_speed_mps",
         lambda d, t: d / t.replace(0, np.nan)),
        (("training_load", "recovery_score"), "load_per_recovery",
         lambda load, rec: load / rec.replace(0, np.nan)),
        (("sprints", "duration_min"), "sprints_per_minute",
         lambda s, t: s / t.replace(0, np.nan)),
        (("heart_rate_avg", "heart_rate_max"), "heart_rate_reserve_proxy",
         lambda avg, mx: mx - avg),
        (("passes_completed", "passes_attempted"), "pass_completion_rate",
         lambda comp, att: comp / att.replace(0, np.nan)),
        (("shots_on_target", "shots"), "shot_accuracy",
         lambda sot, shots: sot / shots.replace(0, np.nan)),
        (("sleep_hours", "training_hours"), "sleep_to_training_ratio",
         lambda sleep, train: sleep / train.replace(0, np.nan)),
    ]
    for sources, target, fn in pairs:
        if target not in out.columns and all(s in lower_to_col for s in sources):
            vals = [pd.to_numeric(out[lower_to_col[s]], errors="coerce") for s in sources]
            out[target] = fn(*vals)
            created.append(target)
    # Generic z-score features for numeric columns; useful for comparative EDA,
    # but deliberately not generated for identifiers or obvious labels.
    numeric_cols = out.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        lc = str(col).lower()
        if any(token in lc for token in ("id", "label", "target", "injury", "risk", "class", "outcome")):
            continue
        target = f"{col}_zscore"
        if target not in out.columns:
            sd = out[col].std()
            if pd.notna(sd) and sd > 0:
                out[target] = (out[col] - out[col].mean()) / sd
                created.append(target)
    return out, created

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
