from __future__ import annotations
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dotenv import load_dotenv
from src.data_utils import read_uploaded_file, dataframe_info, duplicate_column_names, duplicate_content_columns, apply_basic_feature_engineering, to_csv_bytes
from src.ai_utils import ask_gemini, retrieve_sports_notes

load_dotenv()
st.set_page_config(page_title="Sports Analytics Studio | SAS", page_icon="🏟️", layout="wide", initial_sidebar_state="expanded")

ROOT = Path(__file__).parent
SAMPLE_DIR = ROOT / "data" / "sample"

def ensure_samples():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(41)
    n = 1200
    athletes = pd.DataFrame({
        "athlete_id": [f"A{i:04d}" for i in range(n)],
        "athlete_name": [f"Athlete {i:04d}" for i in range(n)],
        "sport": rng.choice(["Football", "Basketball", "Rugby", "Athletics"], n),
        "position": rng.choice(["Forward", "Midfielder", "Defender", "Goalkeeper", "Guard", "Center", "Sprinter"], n),
        "age": rng.integers(16, 36, n),
        "height_cm": rng.normal(178, 10, n).round(1),
        "weight_kg": rng.normal(76, 13, n).round(1),
        "session_date": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
        "training_load": rng.normal(420, 100, n).clip(80, 850).round(1),
        "distance_km": rng.normal(7.5, 2.2, n).clip(1, 15).round(2),
        "duration_min": rng.normal(75, 18, n).clip(20, 130).round(1),
        "top_speed_kmh": rng.normal(29, 4, n).clip(15, 40).round(1),
        "heart_rate_avg": rng.normal(148, 15, n).clip(90, 190).round(0),
        "heart_rate_max": rng.normal(185, 8, n).clip(145, 205).round(0),
        "hrv_ms": rng.normal(58, 15, n).clip(15, 120).round(1),
        "sleep_hours": rng.normal(7.4, 1.1, n).clip(3, 11).round(1),
        "recovery_score": rng.normal(76, 13, n).clip(20, 100).round(0),
        "sprints": rng.poisson(22, n),
        "accel_events": rng.poisson(48, n),
        "decel_events": rng.poisson(43, n),
        "rpe": rng.integers(1, 11, n),
        "performance_score": rng.normal(72, 12, n).clip(20, 100).round(1),
    })
    athletes["avg_speed_kmh"] = (athletes["distance_km"] / (athletes["duration_min"] / 60)).round(2)
    injury_logit = -3.1 + 0.004 * athletes["training_load"] + 0.06 * (athletes["rpe"]) - 0.025 * athletes["recovery_score"] + 0.04 * (8 - athletes["sleep_hours"])
    prob = 1 / (1 + np.exp(-injury_logit))
    athletes["injury_within_7_days"] = (rng.random(n) < prob).astype(int)
    athletes.to_csv(SAMPLE_DIR / "athlete_performance.csv", index=False)

    injury = athletes[["athlete_id", "sport", "position", "age", "training_load", "distance_km", "top_speed_kmh", "heart_rate_avg", "hrv_ms", "sleep_hours", "recovery_score", "rpe", "injury_within_7_days"]].copy()
    injury["acute_load_7d"] = (injury["training_load"] * rng.uniform(4.5, 7.5, n)).round(1)
    injury["chronic_load_28d"] = (injury["training_load"] * rng.uniform(3.2, 5.4, n)).round(1)
    injury["previous_injury"] = rng.binomial(1, 0.22, n)
    injury.to_csv(SAMPLE_DIR / "injury_risk.csv", index=False)

    talent = athletes[["athlete_id", "sport", "position", "age", "height_cm", "weight_kg", "top_speed_kmh", "distance_km", "performance_score", "recovery_score"]].copy()
    talent["coach_rating"] = rng.normal(7.1, 1.2, n).clip(1, 10).round(1)
    talent["technical_score"] = rng.normal(72, 13, n).clip(10, 100).round(1)
    talent["potential_label"] = np.where((talent["performance_score"] > 78) & (talent["technical_score"] > 72), "High", np.where(talent["performance_score"] > 63, "Medium", "Developing"))
    talent.to_csv(SAMPLE_DIR / "talent_identification.csv", index=False)

    events = pd.DataFrame({
        "event_id": [f"E{i:05d}" for i in range(n)],
        "match_date": pd.date_range("2025-02-01", periods=n, freq="D").astype(str),
        "competition": rng.choice(["League", "Cup", "Friendly", "Playoff"], n),
        "home_team": rng.choice(["Falcons", "United", "Rovers", "Athletic", "City"], n),
        "away_team": rng.choice(["Falcons", "United", "Rovers", "Athletic", "City"], n),
        "venue": rng.choice(["Home", "Away", "Neutral"], n),
        "possession_pct": rng.normal(51, 12, n).clip(15, 85).round(1),
        "passes_attempted": rng.integers(180, 800, n),
        "passes_completed": rng.integers(130, 720, n),
        "shots": rng.poisson(12, n),
        "shots_on_target": rng.poisson(5, n),
        "xg": rng.gamma(2.0, 0.7, n).round(2),
        "corners": rng.poisson(5, n),
        "fouls": rng.poisson(11, n),
        "yellow_cards": rng.poisson(1.5, n),
        "red_cards": rng.binomial(1, 0.04, n),
        "goals_for": rng.poisson(1.5, n),
        "goals_against": rng.poisson(1.2, n),
        "attendance": rng.integers(800, 65000, n),
        "temperature_c": rng.normal(19, 7, n).round(1),
        "humidity_pct": rng.integers(25, 96, n),
    })
    events["passes_completed"] = np.minimum(events["passes_completed"], events["passes_attempted"])
    events["shots_on_target"] = np.minimum(events["shots_on_target"], events["shots"])
    events.to_csv(SAMPLE_DIR / "event_analytics.csv", index=False)

    sensor = athletes[["athlete_id", "session_date", "training_load", "distance_km", "top_speed_kmh", "heart_rate_avg", "hrv_ms", "sleep_hours"]].copy()
    idx = rng.choice(n, size=35, replace=False)
    sensor.loc[idx, "top_speed_kmh"] = sensor.loc[idx, "top_speed_kmh"] * rng.uniform(1.8, 2.5, len(idx))
    sensor.loc[rng.choice(n, 25, replace=False), "heart_rate_avg"] = rng.uniform(30, 60, 25)
    sensor.to_csv(SAMPLE_DIR / "sensor_anomalies.csv", index=False)

ensure_samples()

if "original_df" not in st.session_state:
    st.session_state.original_df = None
    st.session_state.active_df = None
    st.session_state.undo_stack = []
    st.session_state.data_name = None
    st.session_state.model_results = None

def set_dataset(df, name):
    st.session_state.original_df = df.copy(deep=True)
    st.session_state.active_df = df.copy(deep=True)
    st.session_state.undo_stack = []
    st.session_state.data_name = name
    st.session_state.model_results = None

def snapshot():
    if st.session_state.active_df is not None:
        st.session_state.undo_stack.append(st.session_state.active_df.copy(deep=True))
        st.session_state.undo_stack = st.session_state.undo_stack[-15:]

def navigate_to(target):
    st.session_state.page_nav = target

def sidebar_nav():
    st.sidebar.markdown("# 🏟️ SAS")
    st.sidebar.caption("Sports Analytics Studio")
    page = st.sidebar.radio("Navigate", [
        "Home", "Data Laboratory", "Athlete Performance", "Injury Risk",
        "Talent Identification", "Event Analytics", "Anomaly Detection", "AI Assistant"
    ], key="page_nav")
    st.sidebar.divider()
    if st.session_state.active_df is not None:
        st.sidebar.success(f"Active data: {st.session_state.data_name}")
        st.sidebar.caption(f"{len(st.session_state.active_df):,} rows × {st.session_state.active_df.shape[1]:,} columns")
        c1, c2 = st.sidebar.columns(2)
        if c1.button("Undo", use_container_width=True, disabled=not st.session_state.undo_stack):
            st.session_state.active_df = st.session_state.undo_stack.pop()
            st.rerun()
        if c2.button("Reset", use_container_width=True):
            st.session_state.active_df = st.session_state.original_df.copy(deep=True)
            st.session_state.undo_stack = []
            st.session_state.model_results = None
            st.rerun()
    st.sidebar.caption("Synthetic examples are for demonstration only.")
    return page

def show_data_summary(df):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

def render_data_laboratory():
    st.title("🧪 Data Laboratory")
    st.write("Upload a dataset or choose a generated sample. Changes are reversible using **Undo**; **Reset** restores the original loaded data.")
    samples = {
        "Athlete performance": "athlete_performance.csv",
        "Injury risk": "injury_risk.csv",
        "Talent identification": "talent_identification.csv",
        "Event analytics": "event_analytics.csv",
        "Sensor anomalies": "sensor_anomalies.csv",
    }
    c1, c2 = st.columns([2, 3])
    with c1:
        chosen = st.selectbox("Load sample dataset", list(samples))
        if st.button("Load selected sample", type="primary"):
            df = pd.read_csv(SAMPLE_DIR / samples[chosen])
            set_dataset(df, samples[chosen])
            st.success("Sample loaded.")
            st.rerun()
    with c2:
        uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        if uploaded is not None and st.button("Load uploaded dataset"):
            try:
                df = read_uploaded_file(uploaded)
                set_dataset(df, uploaded.name)
                st.success(f"Loaded {uploaded.name}")
                st.rerun()
            except Exception as e:
                st.error(f"Could not load file: {e}")

    if st.session_state.active_df is None:
        st.info("Load a sample dataset or upload a file to begin.")
        return
    df = st.session_state.active_df
    show_data_summary(df)
    tabs = st.tabs(["Preview & Info", "Missing Values", "Duplicates", "EDA & Charts", "Feature Engineering", "Export"])
    with tabs[0]:
        st.subheader("Dataset preview")
        st.dataframe(df.head(100), use_container_width=True, height=320)
        a, b = st.columns(2)
        with a:
            st.markdown("**Head**")
            st.dataframe(df.head(int(st.number_input("Head rows", 1, 100, 5, key="head_n"))), use_container_width=True)
        with b:
            st.markdown("**Tail**")
            st.dataframe(df.tail(5), use_container_width=True)
        st.markdown("**Column information**")
        st.dataframe(dataframe_info(df), use_container_width=True)
        st.markdown("**Numeric summary**")
        st.dataframe(df.describe(include="number").T, use_container_width=True)
        st.markdown("**Categorical summary**")
        cats = df.select_dtypes(exclude="number")
        if len(cats.columns):
            st.dataframe(cats.describe().T, use_container_width=True)
    with tabs[1]:
        st.subheader("Missing-value inspection")
        null_info = dataframe_info(df).query("null_count > 0")
        if null_info.empty:
            st.success("No missing values found.")
        else:
            st.dataframe(null_info, use_container_width=True)
            col = st.selectbox("View rows with missing values in column", ["Any column"] + null_info["column"].tolist())
            if col == "Any column":
                st.dataframe(df[df.isna().any(axis=1)].head(500), use_container_width=True)
            else:
                st.dataframe(df[df[col].isna()].head(500), use_container_width=True)
            st.warning("Recommendation: inspect why values are missing before dropping. Imputation may be safer for ML; drop rows only if the loss is small and not systematically biased.")
            action = st.radio("Cleaning action", ["Do nothing", "Drop rows with any null", "Drop rows missing selected column", "Fill numeric with median and categories with mode"], horizontal=False)
            if st.button("Apply missing-value action", disabled=action == "Do nothing"):
                snapshot()
                if action == "Drop rows with any null":
                    st.session_state.active_df = df.dropna().copy()
                elif action == "Drop rows missing selected column":
                    if col == "Any column":
                        st.error("Select a specific column for this action.")
                    else:
                        st.session_state.active_df = df.dropna(subset=[col]).copy()
                else:
                    out = df.copy()
                    for c in out.columns:
                        if out[c].isna().any():
                            if pd.api.types.is_numeric_dtype(out[c]):
                                out[c] = out[c].fillna(out[c].median())
                            else:
                                mode = out[c].mode(dropna=True)
                                if len(mode):
                                    out[c] = out[c].fillna(mode.iloc[0])
                    st.session_state.active_df = out
                st.rerun()
    with tabs[2]:
        st.subheader("Duplicate inspection")
        duplicate_rows = int(df.duplicated().sum())
        duplicate_names = duplicate_column_names(df)
        duplicate_content = duplicate_content_columns(df)
        a, b, c = st.columns(3)
        a.metric("Duplicate rows", duplicate_rows)
        b.metric("Repeated column names", len(duplicate_names))
        c.metric("Identical-content columns", len(duplicate_content))
        if duplicate_rows:
            st.dataframe(df[df.duplicated(keep=False)].head(300), use_container_width=True)
            if st.button("Drop duplicate rows"):
                snapshot()
                st.session_state.active_df = df.drop_duplicates().copy()
                st.rerun()
        if duplicate_names:
            st.warning(f"Repeated column names: {duplicate_names}. Duplicate labels can complicate processing.")
        if duplicate_content:
            st.write("Column positions with identical content (zero-based, later duplicates):", duplicate_content)
            labels = [str(df.columns[i]) for i in duplicate_content]
            st.caption("If column names are repeated, the position list is safer than names.")
            if st.button("Drop identical-content duplicate columns"):
                snapshot()
                st.session_state.active_df = df.iloc[:, [i for i in range(df.shape[1]) if i not in duplicate_content]].copy()
                st.rerun()
        st.info("Do not remove repeated rows automatically if each row represents a legitimate repeated event or sensor reading. Verify identifiers and timestamps first.")
    with tabs[3]:
        st.subheader("Exploratory Data Analysis")
        numeric = df.select_dtypes(include=np.number).columns.tolist()
        all_cols = df.columns.tolist()
        if numeric:
            target_col = st.selectbox("Numeric distribution", numeric)
            st.plotly_chart(px.histogram(df, x=target_col, marginal="box", title=f"Distribution of {target_col}"), use_container_width=True)
            xcol = st.selectbox("X-axis", all_cols, index=0, key="eda_x")
            y_options = [c for c in numeric if c != xcol] or numeric
            ycol = st.selectbox("Y-axis (numeric)", y_options, key="eda_y")
            color_options = ["(none)"] + [c for c in all_cols if c not in (xcol, ycol)]
            color_col = st.selectbox("Color by (optional)", color_options)
            try:
                st.plotly_chart(px.scatter(df, x=xcol, y=ycol, color=None if color_col == "(none)" else color_col, title=f"{ycol} vs {xcol}"), use_container_width=True)
            except Exception as e:
                st.warning(f"Could not plot selected columns: {e}")
            if len(numeric) >= 2:
                st.markdown("**Correlation heatmap (Pearson)**")
                corr = df[numeric].corr(numeric_only=True)
                st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), use_container_width=True)
                st.caption("Correlation describes linear association, not causation. Correlations can be distorted by outliers and missingness.")
        st.markdown("**Missingness overview**")
        missing = df.isna().sum().sort_values(ascending=False)
        if missing.sum():
            st.plotly_chart(px.bar(x=missing.index, y=missing.values, labels={"x":"Column","y":"Missing cells"}), use_container_width=True)
        else:
            st.caption("No missing values.")
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        if cat_cols:
            cat_col = st.selectbox("Category frequency", cat_cols)
            st.plotly_chart(px.bar(df[cat_col].astype(str).value_counts().head(25).reset_index(), x=cat_col, y="count", title=f"Top values in {cat_col}"), use_container_width=True)
    with tabs[4]:
        st.subheader("Feature engineering")
        st.write("Feature engineering can improve predictive signal when derived features reflect the sport and prediction moment. It can also create leakage or redundant columns.")
        st.markdown("- Potential examples: average speed, pass completion rate, shots-on-target rate, sprint rate, load-to-recovery ratio, rolling workload and z-scores.")
        st.markdown("- Avoid deriving features from future outcomes or using post-event data for a pre-event prediction.")
        decision = st.radio("Apply suggested feature engineering?", ["Review only", "Yes, add available derived features"], horizontal=True)
        if st.button("Preview feature engineering"):
            engineered, created = apply_basic_feature_engineering(df)
            st.session_state["feature_preview"] = engineered
            st.session_state["feature_created"] = created
        if "feature_preview" in st.session_state:
            st.write("Features that would be added:", st.session_state.get("feature_created", []))
            st.dataframe(st.session_state["feature_preview"].head(20), use_container_width=True)
            if decision == "Yes, add available derived features" and st.button("Apply feature engineering", type="primary"):
                snapshot()
                st.session_state.active_df = st.session_state["feature_preview"].copy()
                st.session_state.model_results = None
                st.session_state.pop("feature_preview", None)
                st.rerun()
    with tabs[5]:
        st.subheader("Export active dataset")
        st.download_button("Download active dataset as CSV", data=to_csv_bytes(df), file_name="sas_active_dataset.csv", mime="text/csv")
        try:
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="SAS_Data")
            st.download_button("Download active dataset as Excel", data=buffer.getvalue(), file_name="sas_active_dataset.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.caption(f"Excel export unavailable: {e}")
        st.caption("Original data remains available through Reset until a new dataset is loaded.")


def render_athlete_performance():
    st.title("🏃 Athlete Performance")
    st.write("Build a leaderboard from your active dataset. Choose performance metrics and decide whether higher or lower values are better.")
    if st.session_state.active_df is None:
        st.info("Load a dataset in Data Laboratory or load the athlete performance sample below.")
        if st.button("Load athlete performance sample", type="primary"):
            set_dataset(pd.read_csv(SAMPLE_DIR / "athlete_performance.csv"), "athlete_performance.csv")
            st.rerun()
        return

    df = st.session_state.active_df.copy()
    show_data_summary(df)
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric:
        st.warning("This dataset has no numeric columns to rank.")
        return

    likely_ids = [c for c in df.columns if str(c).lower() in ("athlete_id", "athlete_name", "player_id", "player_name", "name")]
    default_id = likely_ids[0] if likely_ids else "(each row is an observation)"
    id_options = ["(each row is an observation)"] + df.columns.tolist()
    athlete_col = st.selectbox("Athlete identifier", id_options,
                               index=id_options.index(default_id) if default_id in id_options else 0)
    exclude = {athlete_col} if athlete_col != "(each row is an observation)" else set()
    metric_options = [c for c in numeric if c not in exclude and not any(t in str(c).lower() for t in ("_id", "id_"))]
    default_metrics = [c for c in ["performance_score", "top_speed_kmh", "distance_km", "recovery_score"] if c in metric_options]
    if not default_metrics:
        default_metrics = metric_options[:min(3, len(metric_options))]
    selected_metrics = st.multiselect("Metrics to include in overall score", metric_options, default=default_metrics)
    if not selected_metrics:
        st.warning("Select at least one numeric performance metric.")
        return

    with st.expander("Configure what counts as good performance", expanded=True):
        st.caption("Higher-is-better is suitable for scores, speed or distance where relevant. Lower-is-better may suit error counts or time-to-completion. Choose based on your sport and metric definitions.")
        higher_defaults = [c for c in selected_metrics if any(t in str(c).lower() for t in ("score", "speed", "distance", "recovery", "completed", "accuracy", "hrv", "jump_height"))]
        higher = st.multiselect("Higher values are better", selected_metrics, default=higher_defaults)
        lower = st.multiselect("Lower values are better", selected_metrics,
                               default=[c for c in selected_metrics if c not in higher_defaults])
        st.caption("Metrics not selected in either direction are excluded from the composite score.")

    active_metrics = [c for c in selected_metrics if c in higher or c in lower]
    if not active_metrics:
        st.warning("Choose at least one metric direction.")
        return

    work = df.copy()
    for c in active_metrics:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    # Aggregate repeated sessions to one row per athlete so prolific athletes are not ranked multiple times.
    if athlete_col != "(each row is an observation)":
        work = work.dropna(subset=[athlete_col])
        agg_map = {c: "mean" for c in active_metrics}
        other_cols = [c for c in work.columns if c not in active_metrics and c != athlete_col]
        for c in other_cols:
            if c not in agg_map:
                agg_map[c] = "first"
        ranked = work.groupby(athlete_col, dropna=False, as_index=False).agg(agg_map)
        ranked = ranked.rename(columns={athlete_col: "Athlete"})
    else:
        ranked = work.copy()
        ranked["Athlete"] = [f"Observation {i+1}" for i in range(len(ranked))]

    # Min-max scale each selected metric to 0-100, reversing metrics where lower is better.
    component_cols = []
    for c in active_metrics:
        vals = pd.to_numeric(ranked[c], errors="coerce")
        lo, hi = vals.min(), vals.max()
        if pd.isna(lo) or pd.isna(hi):
            continue
        if hi == lo:
            scaled = pd.Series(50.0, index=ranked.index)
        else:
            scaled = (vals - lo) / (hi - lo) * 100
        if c in lower:
            scaled = 100 - scaled
        comp = f"{c} (score)"
        ranked[comp] = scaled
        component_cols.append(comp)

    if not component_cols:
        st.error("Selected metrics do not contain usable numeric values.")
        return
    ranked["Overall Score"] = ranked[component_cols].mean(axis=1).round(2)
    ranked["Rank"] = ranked["Overall Score"].rank(method="min", ascending=False).astype("Int64")
    ranked = ranked.sort_values(["Overall Score"], ascending=False, na_position="last")
    top = ranked.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Athletes / observations", f"{len(ranked):,}")
    c2.metric("Top overall score", f"{top['Overall Score']:.1f}/100")
    c3.metric("Top ranked entry", str(top["Athlete"]))

    st.subheader("Performance leaderboard")
    display_cols = ["Rank", "Athlete"] + active_metrics + ["Overall Score"]
    st.dataframe(ranked[display_cols].reset_index(drop=True), use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(ranked.head(20).sort_values("Overall Score"),
                           x="Overall Score", y="Athlete", orientation="h",
                           title="Top 20 athletes by composite score",
                           hover_data=active_metrics), use_container_width=True)

    chosen_athlete = st.selectbox("Inspect an athlete", ranked["Athlete"].astype(str).tolist())
    selected_row = ranked[ranked["Athlete"].astype(str) == chosen_athlete].iloc[0]
    st.markdown(f"**Rank:** {int(selected_row['Rank'])} · **Overall score:** {selected_row['Overall Score']:.1f}/100")
    raw_cols = [c for c in active_metrics if c in ranked.columns]
    st.plotly_chart(px.bar(x=raw_cols, y=[selected_row[c] for c in raw_cols],
                           labels={"x":"Metric", "y":"Observed value"},
                           title=f"Selected athlete: {chosen_athlete}"), use_container_width=True)

    st.download_button("Download leaderboard CSV",
                       data=to_csv_bytes(ranked[display_cols]),
                       file_name="athlete_performance_leaderboard.csv",
                       mime="text/csv")
    st.info("The composite score min-max normalizes selected metrics and gives each selected metric equal weight. It is a configurable exploratory ranking, not a universal definition of athletic quality. Validate metrics and directions with coaches and sport scientists.")
    st.markdown("### Model-based performance analysis")
    st.caption("Use regression to predict a numeric outcome such as performance_score. Model prediction is different from ranking observed performance.")
    model_comparison_ui(df, default_target="performance_score" if "performance_score" in df.columns else None,
                        task="regression", key_prefix="athlete_performance_model")


def render_injury_risk():
    st.title("🛡️ Injury Risk")
    st.write("Rank athletes by estimated injury risk for a selected sport. This is an experimental statistical screening aid, not medical advice or a diagnosis.")
    if st.session_state.active_df is None:
        st.info("Load a dataset in Data Laboratory or load the injury-risk sample below.")
        if st.button("Load injury-risk sample", type="primary"):
            set_dataset(pd.read_csv(SAMPLE_DIR / "injury_risk.csv"), "injury_risk.csv")
            st.rerun()
        return
    df = st.session_state.active_df.copy()
    show_data_summary(df)
    sport_col = next((c for c in df.columns if str(c).lower() in ("sport", "sports", "sport_type")), None)
    athlete_col = next((c for c in df.columns if str(c).lower() in ("athlete_id", "athlete_name", "player_id", "player_name", "athlete", "player")), None)
    if sport_col:
        sports = sorted(df[sport_col].dropna().astype(str).unique().tolist())
        selected_sport = st.selectbox("Select sport", ["All sports"] + sports, key="injury_sport")
        filtered = df if selected_sport == "All sports" else df[df[sport_col].astype(str) == selected_sport]
    else:
        st.warning("No sport column was found. Add a 'sport' column to filter by sport.")
        filtered = df
    n = st.number_input("Number of athletes to display (N)", min_value=1, max_value=100, value=10, step=1, key="injury_top_n")
    target_candidates = [c for c in filtered.columns if any(t in str(c).lower() for t in ("injury_within", "injury_risk", "injury_label", "injured", "injury_next", "injury"))]
    target_candidates = [c for c in target_candidates if filtered[c].nunique(dropna=True) <= 20]
    default_target = next((c for c in target_candidates if "injury_within_7_days" in str(c).lower()), target_candidates[0] if target_candidates else None)
    st.markdown("### Risk ranking")
    st.caption("If a historical injury label is available, SAS trains a model and uses predicted probability. Otherwise, it can rank using a transparent workload/recovery proxy, clearly labeled as a heuristic.")
    if default_target:
        target = st.selectbox("Historical injury outcome label", target_candidates, index=target_candidates.index(default_target), key="injury_target")
        feature_candidates = [c for c in filtered.columns if c != target and c != athlete_col and c != sport_col]
        numeric_features = filtered[feature_candidates].select_dtypes(include=np.number).columns.tolist()
        safe_defaults = [c for c in ["training_load", "acute_load_7d", "chronic_load_28d", "previous_injury", "rpe", "recovery_score", "sleep_hours", "hrv_ms", "age"] if c in numeric_features]
        features = st.multiselect("Features available at prediction time", numeric_features, default=safe_defaults, key="injury_features")
        if st.button("Estimate injury risk", type="primary"):
            try:
                from sklearn.compose import ColumnTransformer
                from sklearn.pipeline import Pipeline
                from sklearn.impute import SimpleImputer
                from sklearn.preprocessing import StandardScaler
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import roc_auc_score
                if len(features) < 1:
                    raise ValueError("Select at least one numeric feature.")
                train = filtered.dropna(subset=[target]).copy()
                y = train[target].astype(str)
                if y.nunique() < 2 or len(train) < 30:
                    raise ValueError("Need at least 30 rows and at least two outcome classes.")
                X = train[features].apply(pd.to_numeric, errors="coerce")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42,
                    stratify=y if y.value_counts().min() >= 2 else None
                )
                model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("classifier", RandomForestClassifier(n_estimators=250, class_weight=None, random_state=42, n_jobs=-1))
                ])
                model.fit(X_train, y_train)
                proba = model.predict_proba(X)
                classes = list(model.named_steps["classifier"].classes_)
                positive_label = next((c for c in classes if str(c).lower() in ("1", "true", "yes", "injury", "injured", "high")), classes[-1])
                positive_idx = classes.index(positive_label)
                # Do not use the outcome column as a feature; this estimates risk for current rows.
                output = train.copy()
                output["estimated_injury_risk_pct"] = proba[:, positive_idx] * 100
                output["risk_group"] = pd.cut(output["estimated_injury_risk_pct"], bins=[-0.001, 33.333, 66.667, 100.001], labels=["Lower model score", "Moderate model score", "Higher model score"])
                st.session_state.injury_risk_output = output
                st.session_state.injury_risk_model_features = features
                st.session_state.injury_risk_target = target
                st.session_state.injury_risk_positive_label = str(positive_label)
            except Exception as e:
                st.warning(f"Supervised risk estimation could not run ({e}). Showing an exploratory workload/recovery heuristic ranking instead, so you can still inspect the top and bottom athletes.")
                # Fallback ranking: never leave the athlete tables empty just because model fitting fails.
                fallback = filtered.copy()
                fallback_score = pd.Series(0.0, index=fallback.index, dtype=float)
                used = 0
                fallback_features = [c for c in features if c in fallback.columns] if "features" in locals() else []
                for c in fallback_features:
                    values = pd.to_numeric(fallback[c], errors="coerce")
                    if values.notna().sum() < 2 or values.nunique(dropna=True) < 2:
                        continue
                    std = values.std()
                    if pd.isna(std) or std == 0:
                        continue
                    z = ((values - values.mean()) / std).fillna(0)
                    low_is_risk = any(t in str(c).lower() for t in ("recovery", "sleep", "hrv"))
                    fallback_score += -z if low_is_risk else z
                    used += 1
                if used == 0:
                    # Use common workload/recovery columns even if feature selection was empty.
                    for c in ["training_load", "acute_load_7d", "previous_injury", "rpe", "recovery_score", "sleep_hours", "hrv_ms"]:
                        if c not in fallback.columns:
                            continue
                        values = pd.to_numeric(fallback[c], errors="coerce")
                        if values.notna().sum() < 2 or values.nunique(dropna=True) < 2:
                            continue
                        std = values.std()
                        if pd.isna(std) or std == 0:
                            continue
                        z = ((values - values.mean()) / std).fillna(0)
                        low_is_risk = any(t in c.lower() for t in ("recovery", "sleep", "hrv"))
                        fallback_score += -z if low_is_risk else z
                        used += 1
                if used:
                    score_range = fallback_score.max() - fallback_score.min()
                    fallback["estimated_injury_risk_pct"] = (100 * (fallback_score - fallback_score.min()) / score_range).clip(0, 100) if score_range > 0 else 50.0
                    fallback["risk_group"] = "Heuristic only — model fallback"
                    st.session_state.injury_risk_output = fallback
                else:
                    st.error("Could not calculate a fallback score. Select numeric workload/recovery features or check your dataset.")
    else:
        st.info("No clear historical injury outcome column was detected. You can use a heuristic ranking, but it is not a trained probability model.")
        numeric = filtered.select_dtypes(include=np.number).columns.tolist()
        risk_features = [c for c in ["training_load", "acute_load_7d", "previous_injury", "rpe", "recovery_score", "sleep_hours", "hrv_ms"] if c in numeric]
        if st.button("Build heuristic risk ranking", type="primary"):
            out = filtered.copy()
            score = pd.Series(0.0, index=out.index)
            used = 0
            for c in risk_features:
                x = pd.to_numeric(out[c], errors="coerce")
                if x.notna().sum() < 2 or x.max() == x.min():
                    continue
                z = (x - x.mean()) / (x.std() or 1)
                low_is_risk = any(t in c.lower() for t in ("recovery", "sleep", "hrv"))
                score += (-z if low_is_risk else z).fillna(0)
                used += 1
            if not used:
                st.error("No supported workload/recovery features found.")
            else:
                out["estimated_injury_risk_pct"] = (100 * (score - score.min()) / (score.max() - score.min() + 1e-9)).clip(0,100)
                out["risk_group"] = "Heuristic only"
                st.session_state.injury_risk_output = out

    if "injury_risk_output" in st.session_state:
        out = st.session_state.injury_risk_output
        if sport_col and "All sports" not in [st.session_state.get("injury_sport")]:
            # Output was created from the sport-filtered data, so it already reflects the selected sport.
            pass
        if athlete_col:
            # Collapse multiple records to athlete-level mean estimated risk.
            group_cols = [athlete_col] + ([sport_col] if sport_col else [])
            agg = {"estimated_injury_risk_pct": "mean"}
            for c in ["risk_group", "position", "age", "previous_injury"]:
                if c in out.columns and c not in group_cols:
                    agg[c] = "first"
            leaderboard = out.groupby(group_cols, dropna=False, as_index=False).agg(agg)
        else:
            leaderboard = out.copy()
            leaderboard["athlete_record"] = [f"Record {i+1}" for i in range(len(leaderboard))]
            athlete_col = "athlete_record"
        leaderboard = leaderboard.sort_values("estimated_injury_risk_pct", ascending=False)
        left, right = st.columns(2)
        with left:
            st.subheader(f"Top {int(n)} higher model-score athletes")
            st.dataframe(leaderboard.head(int(n)), use_container_width=True, hide_index=True)
        with right:
            st.subheader(f"Top {int(n)} lower model-score athletes")
            st.dataframe(leaderboard.tail(int(n)).sort_values("estimated_injury_risk_pct"), use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(leaderboard.head(int(n)), x=athlete_col, y="estimated_injury_risk_pct",
                               color="risk_group" if "risk_group" in leaderboard.columns else None,
                               title=f"Higher estimated risk — top {int(n)}",
                               labels={"estimated_injury_risk_pct":"Estimated model score (%)"}), use_container_width=True)
        st.download_button("Download injury risk rankings", to_csv_bytes(leaderboard), "injury_risk_rankings.csv", "text/csv")
        st.warning("A predicted score is not a clinical risk assessment. Model rankings can be biased or miscalibrated, especially with synthetic, small, imbalanced, or time-leaky data. Have qualified sports medicine staff interpret findings.")

    st.markdown("### Optional model comparison")
    model_comparison_ui(filtered, default_target=default_target, task="classification", key_prefix="injury_model")


def render_talent_identification():
    st.title("🏆 Talent Identification")
    st.write("Explore athlete potential within a selected sport and create a top-N talent leaderboard using selected attributes.")
    if st.session_state.active_df is None:
        st.info("Load a dataset in Data Laboratory or load the talent identification sample below.")
        if st.button("Load talent identification sample", type="primary"):
            set_dataset(pd.read_csv(SAMPLE_DIR / "talent_identification.csv"), "talent_identification.csv")
            st.rerun()
        return
    df = st.session_state.active_df.copy()
    show_data_summary(df)
    sport_col = next((c for c in df.columns if str(c).lower() in ("sport", "sports", "sport_type")), None)
    athlete_col = next((c for c in df.columns if str(c).lower() in ("athlete_id", "athlete_name", "player_id", "player_name", "athlete", "player", "name")), None)
    if sport_col:
        sports = sorted(df[sport_col].dropna().astype(str).unique().tolist())
        selected_sport = st.selectbox("Select sport", ["All sports"] + sports, key="talent_sport")
        filtered = df if selected_sport == "All sports" else df[df[sport_col].astype(str) == selected_sport]
    else:
        st.warning("No sport column was found. Add a 'sport' column to enable sport-specific rankings.")
        filtered = df
    n = st.number_input("Number of talents to display (N)", min_value=1, max_value=100, value=10, step=1, key="talent_top_n")
    numeric = filtered.select_dtypes(include=np.number).columns.tolist()
    excluded = {athlete_col, sport_col}
    metric_candidates = [c for c in numeric if c not in excluded and not any(t in str(c).lower() for t in ("_id", "id_"))]
    defaults = [c for c in ["performance_score", "technical_score", "coach_rating", "top_speed_kmh", "recovery_score"] if c in metric_candidates]
    if not defaults:
        defaults = metric_candidates[:min(3, len(metric_candidates))]
    selected_metrics = st.multiselect("Talent assessment metrics", metric_candidates, default=defaults, key="talent_metrics")
    with st.expander("Metric direction and weighting", expanded=True):
        higher_default = [c for c in selected_metrics if any(t in str(c).lower() for t in ("score", "rating", "speed", "distance", "accuracy", "recovery", "jump", "strength"))]
        higher = st.multiselect("Higher values are better", selected_metrics, default=higher_default, key="talent_higher")
        lower = st.multiselect("Lower values are better", selected_metrics, default=[c for c in selected_metrics if c not in higher_default], key="talent_lower")
        weighting = st.selectbox("Metric weighting", ["Equal weight", "Emphasize first selected metric"], key="talent_weighting")
    if not selected_metrics:
        st.warning("Select at least one talent metric.")
        return
    if st.button("Identify top talents", type="primary"):
        try:
            work = filtered.copy()
            for c in selected_metrics:
                work[c] = pd.to_numeric(work[c], errors="coerce")
            if athlete_col:
                group_cols = [athlete_col] + ([sport_col] if sport_col else [])
                agg = {c: "mean" for c in selected_metrics}
                for c in work.columns:
                    if c not in group_cols and c not in agg:
                        agg[c] = "first"
                ranked = work.groupby(group_cols, dropna=False, as_index=False).agg(agg)
                ranked = ranked.rename(columns={athlete_col: "Athlete"})
            else:
                ranked = work.copy()
                ranked["Athlete"] = [f"Record {i+1}" for i in range(len(ranked))]
            score_cols = []
            for c in selected_metrics:
                x = pd.to_numeric(ranked[c], errors="coerce")
                lo, hi = x.min(), x.max()
                if pd.isna(lo) or pd.isna(hi):
                    continue
                scaled = pd.Series(50.0, index=ranked.index) if hi == lo else (x - lo) / (hi - lo) * 100
                if c in lower:
                    scaled = 100 - scaled
                score_name = f"{c}_normalized"
                ranked[score_name] = scaled
                score_cols.append(score_name)
            if not score_cols:
                raise ValueError("No selected metrics contain usable numeric values.")
            if weighting == "Emphasize first selected metric":
                first_score = f"{selected_metrics[0]}_normalized"
                ranked["Talent Score"] = ranked[score_cols].mean(axis=1) * 0.5 + ranked[first_score] * 0.5
            else:
                ranked["Talent Score"] = ranked[score_cols].mean(axis=1)
            ranked["Talent Score"] = ranked["Talent Score"].round(2)
            ranked["Rank"] = ranked["Talent Score"].rank(method="min", ascending=False).astype("Int64")
            ranked = ranked.sort_values("Talent Score", ascending=False)
            st.session_state.talent_ranked_output = ranked
            st.session_state.talent_ranked_metrics = selected_metrics
        except Exception as e:
            st.error(f"Talent ranking failed: {e}")
    if "talent_ranked_output" in st.session_state:
        ranked = st.session_state.talent_ranked_output
        top = ranked.head(int(n))
        st.metric(f"Top {int(n)} list", f"{len(top)} athletes")
        show_cols = ["Rank", "Athlete"] + ([sport_col] if sport_col and sport_col in ranked.columns else []) + selected_metrics + ["Talent Score"]
        show_cols = list(dict.fromkeys(c for c in show_cols if c in ranked.columns))
        st.subheader(f"Top {int(n)} talents")
        st.dataframe(top[show_cols], use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(top.sort_values("Talent Score"), x="Talent Score", y="Athlete", orientation="h",
                               title=f"Top {int(n)} talent candidates"), use_container_width=True)
        st.download_button("Download talent leaderboard", to_csv_bytes(ranked[show_cols]), "talent_leaderboard.csv", "text/csv")
        st.info("This is a configurable screening score, not a definitive scouting decision. Use role-specific metrics and contextual factors such as age, biological maturation, competition level, minutes played, and opportunity. Avoid cross-sport comparisons unless the metrics have been appropriately standardized.")
    st.markdown("### Optional potential-label model")
    label = next((c for c in filtered.columns if str(c).lower() in ("potential_label", "talent_label", "scouting_grade")), None)
    if label:
        model_comparison_ui(filtered, default_target=label, task="classification", key_prefix="talent_supervised")

def model_comparison_ui(df, default_target=None, task="classification", key_prefix="model"):
    st.subheader("Train and compare ML models")
    candidates = df.columns.tolist()
    if not candidates:
        st.warning("No columns available.")
        return
    default_idx = candidates.index(default_target) if default_target in candidates else len(candidates)-1
    target = st.selectbox("Target column", candidates, index=default_idx, key=f"{key_prefix}_target")
    options = (["Logistic Regression", "Decision Tree", "Random Forest", "Extra Trees", "Gradient Boosting", "K-Nearest Neighbors", "Support Vector Machine"]
               if task == "classification" else ["Ridge Regression", "Decision Tree", "Random Forest", "Extra Trees", "Gradient Boosting"])
    selected = st.multiselect("Models to compare", options, default=options[:4], key=f"{key_prefix}_models")
    test_size = st.slider("Test set proportion", 0.1, 0.4, 0.2, 0.05, key=f"{key_prefix}_test")
    if st.button("Run selected models", type="primary", key=f"{key_prefix}_run"):
        try:
            from src.ml_utils import compare_models
            with st.spinner("Training models..."):
                results, fitted = compare_models(df, target, task, selected, test_size=test_size)
            st.session_state.model_results = results
            st.session_state.model_fitted = fitted
            st.session_state.model_task = task
            st.session_state.model_target = target
        except Exception as e:
            st.error(f"Model run failed: {e}")
    results = st.session_state.get("model_results")
    if results is not None:
        st.markdown("**Model comparison results**")
        st.dataframe(results, use_container_width=True)
        numeric_metrics = [c for c in results.select_dtypes(include="number").columns if c != "Model"]
        if numeric_metrics:
            metric = st.selectbox("Metric to compare", numeric_metrics, key=f"{key_prefix}_metric")
            st.plotly_chart(px.bar(results, x="Model", y=metric, title=f"Model comparison — {metric}"), use_container_width=True)
        st.info("For classification, larger Accuracy/F1/ROC-AUC is often preferred; for regression, lower MAE/RMSE and higher R² are typical goals. Consider class balance, calibration, interpretability, and validation stability—not just one score.")

def task_page(title, emoji, description, sample_name, task_kind, default_target=None):
    st.title(f"{emoji} {title}")
    st.write(description)
    if st.session_state.active_df is None:
        st.info("No active dataset. Load a dataset in Data Laboratory or use the suggested sample below.")
        if st.button(f"Load {sample_name} sample", type="primary"):
            set_dataset(pd.read_csv(SAMPLE_DIR / sample_name), sample_name)
            st.rerun()
        return
    df = st.session_state.active_df
    show_data_summary(df)
    st.markdown("**Active dataset preview**")
    st.dataframe(df.head(12), use_container_width=True)
    if task_kind in ("classification", "regression"):
        model_comparison_ui(df, default_target=default_target, task=task_kind, key_prefix=title.lower().replace(" ", "_"))
    elif task_kind == "talent":
        numeric = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric) >= 2:
            features = st.multiselect("Select talent attributes", numeric, default=numeric[:min(4, len(numeric))])
            clusters = st.slider("Talent profile clusters", 2, 8, 3)
            if st.button("Identify athlete profiles", type="primary"):
                try:
                    from src.ml_utils import run_kmeans
                    if len(features) < 2:
                        st.error("Select at least two numeric attributes.")
                    else:
                        out = df.copy()
                        out["talent_cluster"] = run_kmeans(out, features, clusters)
                        st.session_state.talent_output = out
                except Exception as e:
                    st.error(str(e))
            if "talent_output" in st.session_state:
                out = st.session_state.talent_output
                st.dataframe(out.head(100), use_container_width=True)
                if len(features) >= 2:
                    st.plotly_chart(px.scatter(out, x=features[0], y=features[1], color="talent_cluster", hover_data=[c for c in ["athlete_id", "athlete_name", "position"] if c in out.columns]), use_container_width=True)
                st.download_button("Download talent profiles", to_csv_bytes(out), "talent_profiles.csv", "text/csv")
        if default_target in df.columns:
            st.markdown("**Optional supervised talent label prediction**")
            model_comparison_ui(df, default_target=default_target, task="classification", key_prefix="talent_supervised")
    elif task_kind == "event":
        numeric = df.select_dtypes(include=np.number).columns.tolist()
        cats = df.select_dtypes(exclude=np.number).columns.tolist()
        if numeric:
            metric = st.selectbox("Event metric", numeric, key="event_metric")
            if cats:
                group = st.selectbox("Group by", cats, key="event_group")
                agg = df.groupby(group, dropna=False)[metric].mean().sort_values(ascending=False).reset_index()
                st.plotly_chart(px.bar(agg.head(30), x=group, y=metric, title=f"Average {metric} by {group}"), use_container_width=True)
            st.plotly_chart(px.histogram(df, x=metric, title=f"Distribution of {metric}"), use_container_width=True)
        st.markdown("**Optional outcome prediction**")
        model_comparison_ui(df, default_target=default_target, task="regression", key_prefix="event_model")
    elif task_kind == "anomaly":
        from src.ml_utils import detect_anomalies
        numeric = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric) >= 2:
            features = st.multiselect("Sensor / performance features", numeric, default=numeric[:min(5, len(numeric))], key="anomaly_features")
            contamination = st.slider("Expected anomalous fraction", 0.01, 0.20, 0.05, 0.01)
            if st.button("Detect anomalies", type="primary"):
                try:
                    if len(features) < 2:
                        st.error("Select at least two numeric features.")
                    else:
                        out, _ = detect_anomalies(df, features, contamination)
                        st.session_state.anomaly_output = out
                except Exception as e:
                    st.error(f"Anomaly detection failed: {e}")
            if "anomaly_output" in st.session_state:
                out = st.session_state.anomaly_output
                count = int((out["anomaly_flag"] == "Anomaly").sum())
                st.metric("Flagged observations", f"{count:,}", f"{count / max(len(out), 1):.1%} of rows")
                st.dataframe(out[out["anomaly_flag"] == "Anomaly"].head(500), use_container_width=True)
                st.download_button("Download anomaly results", to_csv_bytes(out), "anomaly_results.csv", "text/csv")
                st.info("Review flagged rows in context. Sensor errors, rare but valid performance, and real events can all look anomalous.")

page = sidebar_nav()
if page == "Home":
    st.title("🏟️ Sports Analytics Studio")
    st.subheader("Turn sports data into actionable, evidence-aware insights.")
    st.write("An extensible sports analytics workspace for athlete performance, injury-risk research, talent identification, event analysis, and sensor anomaly detection.")
    st.markdown("### Start here")
    cols = st.columns(3)
    cards = [
        ("🧪 Data Laboratory", "Upload and clean data, inspect nulls/duplicates, engineer features and visualize.", "Data Laboratory"),
        ("🏃 Athlete Performance", "Explore workload, speed, recovery and performance models.", "Athlete Performance"),
        ("🛡️ Injury Risk", "Compare experimental classifiers using historical injury labels.", "Injury Risk"),
        ("🏆 Talent Identification", "Discover athlete profiles with clustering and optional classification.", "Talent Identification"),
        ("📊 Event Analytics", "Explore team/event statistics and regression experiments.", "Event Analytics"),
        ("🔎 Anomaly Detection", "Flag unusual sensor or performance observations.", "Anomaly Detection"),
    ]
    for i, (name, desc, target) in enumerate(cards):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"#### {name}")
                st.write(desc)
                st.button("Open", key=f"home_{i}", use_container_width=True,
                          on_click=navigate_to, args=(target,))
    st.markdown("### Recommended workflow")
    st.markdown("1. Load a synthetic sample or upload a CSV/Excel file.\n2. Inspect schema, nulls, duplicates, distributions and potential leakage.\n3. Apply only justified cleaning and feature engineering; Undo or Reset when needed.\n4. Select the relevant task, target, and models, then run the experiment.\n5. Compare metrics, inspect errors, export results, and document assumptions.")
    st.warning("This starter application is for analytics prototyping. Injury-risk outputs are not clinical advice, and synthetic-data model scores do not demonstrate real-world performance.")
elif page == "Data Laboratory":
    render_data_laboratory()
elif page == "Athlete Performance":
    render_athlete_performance()
elif page == "Injury Risk":
    render_injury_risk()
elif page == "Talent Identification":
    render_talent_identification()
elif page == "Event Analytics":
    task_page("Event Analytics", "📊", "Explore event-level statistics by team, competition, venue, or match date. Confirm the unit of observation before modeling.", "event_analytics.csv", "event", "goals_for")
elif page == "Anomaly Detection":
    task_page("Anomaly Detection", "🔎", "Use Isolation Forest to flag unusual numeric sensor or performance patterns. Anomalies require human review and are not automatically faults.", "sensor_anomalies.csv", "anomaly")
elif page == "AI Assistant":
    st.title("✨ AI Assistant")
    st.write("Ask questions about the active dataset. LangChain uses HuggingFaceEmbeddings with sentence-transformers/all-MiniLM-L6-v2 for ChromaDB semantic retrieval; Gemini generates the response when configured.")
    df = st.session_state.active_df
    if df is None:
        st.info("Load a dataset in Data Laboratory first. You can still ask general sports analytics questions.")
        summary = "No active dataset is loaded."
    else:
        summary = f"Dataset: {st.session_state.data_name}\nShape: {df.shape}\nColumns and types:\n{dataframe_info(df).to_string(index=False)}\nSample:\n{df.head(5).to_string(index=False)}"
        with st.expander("Dataset context sent with your question"):
            st.code(summary[:7000])
    question = st.text_area("Your question", placeholder="Which columns need data-quality checks? What leakage risks should I inspect before training injury-risk models?")
    if st.button("Ask AI", type="primary", disabled=not question.strip()):
        notes = retrieve_sports_notes(question)
        st.markdown("**Relevant knowledge retrieved with Hugging Face embeddings**")
        st.caption('Embedding model: sentence-transformers/all-MiniLM-L6-v2')
        for note in notes:
            st.markdown(f"- {note}")
        try:
            with st.spinner("Asking Gemini..."):
                answer = ask_gemini(question + "\n\nRelevant sports analytics notes:\n" + "\n".join(notes), summary)
            st.markdown("### Gemini response")
            st.write(answer)
        except Exception as e:
            st.error(str(e))
    st.caption("Only a limited dataset summary and sample are sent to Gemini. Do not upload or transmit sensitive personal or medical data without appropriate authorization.")
