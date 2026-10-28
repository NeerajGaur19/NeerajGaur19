from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

def _prepare_xy(df, target, task, drop_cols=None):
    data = df.copy()
    drop_cols = drop_cols or []
    data = data.dropna(subset=[target])
    y = data[target]
    X = data.drop(columns=[target] + [c for c in drop_cols if c in data.columns], errors="ignore")
    # Avoid likely unique identifiers that can cause leakage; user can explicitly remove them beforehand.
    for c in list(X.columns):
        if str(c).lower() in ("id", "athlete_id", "player_id", "event_id", "record_id"):
            X = X.drop(columns=c)
    if task == "classification":
        y = y.astype(str)
        if y.nunique() < 2:
            raise ValueError("Classification requires at least two target classes.")
    else:
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid]
    if len(X) < 20:
        raise ValueError("At least 20 valid rows are recommended for a demonstration model.")
    # Remove columns with all values missing.
    X = X.dropna(axis=1, how="all")
    return X, y

def _pipeline(X, estimator):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols))
    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("preprocess", pre), ("model", estimator)])

def compare_models(df, target, task, selected_models, test_size=0.2, random_state=42):
    X, y = _prepare_xy(df, target, task)
    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    if task == "classification":
        model_map = {
            "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
            "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=random_state),
            "Random Forest": RandomForestClassifier(n_estimators=180, class_weight="balanced", random_state=random_state, n_jobs=-1),
            "Extra Trees": ExtraTreesClassifier(n_estimators=180, class_weight="balanced", random_state=random_state, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Support Vector Machine": SVC(probability=True, class_weight="balanced", random_state=random_state),
        }
    else:
        model_map = {
            "Ridge Regression": Ridge(),
            "Decision Tree": __import__("sklearn.tree", fromlist=["DecisionTreeRegressor"]).DecisionTreeRegressor(max_depth=8, random_state=random_state),
            "Random Forest": RandomForestRegressor(n_estimators=180, random_state=random_state, n_jobs=-1),
            "Extra Trees": ExtraTreesRegressor(n_estimators=180, random_state=random_state, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
        }
    results, fitted = [], {}
    for name in selected_models:
        if name not in model_map:
            continue
        try:
            pipe = _pipeline(X_train, model_map[name])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            row = {"Model": name}
            if task == "classification":
                row.update({
                    "Accuracy": accuracy_score(y_test, pred),
                    "Precision (weighted)": precision_score(y_test, pred, average="weighted", zero_division=0),
                    "Recall (weighted)": recall_score(y_test, pred, average="weighted", zero_division=0),
                    "F1 (weighted)": f1_score(y_test, pred, average="weighted", zero_division=0),
                })
                try:
                    if hasattr(pipe, "predict_proba") and y_test.nunique() == 2:
                        row["ROC-AUC"] = roc_auc_score((y_test == sorted(y_test.unique())[-1]).astype(int), pipe.predict_proba(X_test)[:, 1])
                except Exception:
                    pass
            else:
                row.update({
                    "MAE": mean_absolute_error(y_test, pred),
                    "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                    "R²": r2_score(y_test, pred),
                })
            results.append(row)
            fitted[name] = pipe
        except Exception as e:
            results.append({"Model": name, "Error": str(e)[:220]})
    if not results:
        raise ValueError("No selected models could be trained.")
    return pd.DataFrame(results), fitted

def run_kmeans(df, features, n_clusters=3):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    X = df[features].apply(pd.to_numeric, errors="coerce")
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=features, index=df.index)
    Xs = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(Xs)
    return labels

def detect_anomalies(df, features, contamination=0.05):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    X = df[features].apply(pd.to_numeric, errors="coerce")
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=features, index=df.index)
    Xs = StandardScaler().fit_transform(X)
    model = IsolationForest(contamination=contamination, random_state=42)
    pred = model.fit_predict(Xs)
    score = model.decision_function(Xs)
    out = df.copy()
    out["anomaly_flag"] = np.where(pred == -1, "Anomaly", "Normal")
    out["anomaly_score"] = score
    return out, model
