"""Train five classification models on the UCI Bank Marketing dataset.

Run this script in BITS Virtual Lab or another Python environment with internet
access. It downloads the official UCI archive, excludes `duration` to avoid
post-call information leakage, creates a stratified holdout test set, trains
five models, and writes model and evaluation artifacts.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
RANDOM_STATE = 42
TARGET = "y"


def download_dataset() -> pd.DataFrame:
    """Download and read the full 20-input UCI bank-additional dataset."""
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(BytesIO(archive_bytes)) as outer_zip:
        inner_name = next((n for n in outer_zip.namelist()
                           if n.lower().endswith("bank-additional.zip")), None)
        if inner_name is None:
            raise FileNotFoundError("bank-additional.zip was not found in the UCI archive.")
        with zipfile.ZipFile(BytesIO(outer_zip.read(inner_name))) as inner_zip:
            csv_name = next((n for n in inner_zip.namelist()
                             if n.lower().endswith("bank-additional-full.csv")), None)
            if csv_name is None:
                raise FileNotFoundError("bank-additional-full.csv was not found.")
            with inner_zip.open(csv_name) as csv_file:
                return pd.read_csv(csv_file, sep=";")


def make_preprocessor(X: pd.DataFrame, dense: bool = False) -> ColumnTransformer:
    """Build leakage-safe numeric and categorical preprocessing."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])


def score_model(model, X_test, y_test) -> tuple[dict, np.ndarray]:
    """Calculate the six metrics required by the assignment."""
    predictions = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        positive_scores = model.predict_proba(X_test)[:, 1]
    else:
        positive_scores = model.decision_function(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, positive_scores),
        "Precision": precision_score(y_test, predictions, pos_label="yes", zero_division=0),
        "Recall": recall_score(y_test, predictions, pos_label="yes", zero_division=0),
        "F1": f1_score(y_test, predictions, pos_label="yes", zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }
    return metrics, confusion_matrix(y_test, predictions, labels=["no", "yes"])


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    print("Downloading the official UCI Bank Marketing dataset...")
    data = download_dataset()
    if TARGET not in data.columns:
        raise ValueError("Expected target column 'y' was not found.")
    # Exclude duration: UCI notes that it is only known after the call ends.
    X = data.drop(columns=[TARGET, "duration"], errors="ignore")
    y = data[TARGET].astype(str).str.strip().str.lower()
    if not set(y.unique()).issubset({"yes", "no"}):
        raise ValueError(f"Unexpected target labels: {sorted(y.unique())}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    # Keep the test CSV in the exact format expected by the deployed app.
    test_frame = X_test.copy()
    test_frame[TARGET] = y_test.values
    test_frame.to_csv(ROOT / "test_data.csv", index=False)

    model_specs = {
        "Logistic Regression": ("logistic_regression.joblib", LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE)),
        "Decision Tree": ("decision_tree.joblib", DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=8, class_weight="balanced", random_state=RANDOM_STATE)),
        "KNN": ("knn.joblib", KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1)),
        "Naive Bayes": ("naive_bayes.joblib", GaussianNB()),
        "Support Vector Machine": ("svm.joblib", LinearSVC(
            class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000)),
    }
    rows = []
    for display_name, (filename, estimator) in model_specs.items():
        print(f"Training {display_name}...")
        # GaussianNB requires dense input; other models retain sparse one-hot data.
        preprocessor = make_preprocessor(X_train, dense=(display_name == "Naive Bayes"))
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
        pipeline.fit(X_train, y_train)
        metrics, matrix = score_model(pipeline, X_test, y_test)
        joblib.dump(pipeline, MODEL_DIR / filename)
        row = {"Model": display_name, **metrics}
        rows.append(row)
        pd.DataFrame(matrix, index=["Actual no", "Actual yes"],
                     columns=["Predicted no", "Predicted yes"]).to_csv(
                         REPORT_DIR / f"{filename.replace('.joblib', '')}_confusion_matrix.csv")
        print("  " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    results = pd.DataFrame(rows)
    results.to_csv(REPORT_DIR / "model_metrics.csv", index=False)
    print("\nTraining complete.")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved five models in: {MODEL_DIR}")
    print(f"Saved holdout test data in: {ROOT / 'test_data.csv'}")

if __name__ == "__main__":
    main()
