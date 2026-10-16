"""
train.py - Customer Churn Prediction (Telco Churn structure)

What it does:
  1. Loads the Telco churn CSV if present, otherwise generates synthetic data
     with the same schema.
  2. Cleans missing values.
  3. Encodes categorical variables and scales numerics.
  4. Trains an XGBoost classifier with a proper train/test split.
  5. Prints the classification report and ROC-AUC.
  6. Saves model + scaler + feature metadata into ./models

Usage:
    python train.py
    python train.py --data path/to/telco_churn.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
MODELS_DIR = Path(__file__).resolve().parent / "models"

CATEGORICAL = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
def generate_synthetic(n_rows: int = 3000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Build a synthetic dataset shaped like the Telco Customer Churn data."""
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 73, n_rows)
    monthly = np.round(rng.uniform(18, 120, n_rows), 2)
    total = np.round(monthly * tenure * rng.uniform(0.9, 1.05, n_rows), 2)

    df = pd.DataFrame(
        {
            "customerID": [f"SYN-{i:05d}" for i in range(n_rows)],
            "gender": rng.choice(["Male", "Female"], n_rows),
            "SeniorCitizen": rng.choice([0, 1], n_rows, p=[0.84, 0.16]),
            "Partner": rng.choice(["Yes", "No"], n_rows),
            "Dependents": rng.choice(["Yes", "No"], n_rows, p=[0.3, 0.7]),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], n_rows, p=[0.9, 0.1]),
            "MultipleLines": rng.choice(
                ["Yes", "No", "No phone service"], n_rows, p=[0.42, 0.48, 0.10]
            ),
            "InternetService": rng.choice(
                ["DSL", "Fiber optic", "No"], n_rows, p=[0.34, 0.44, 0.22]
            ),
            "OnlineSecurity": rng.choice(
                ["Yes", "No", "No internet service"], n_rows, p=[0.29, 0.49, 0.22]
            ),
            "OnlineBackup": rng.choice(
                ["Yes", "No", "No internet service"], n_rows, p=[0.34, 0.44, 0.22]
            ),
            "DeviceProtection": rng.choice(
                ["Yes", "No", "No internet service"], n_rows, p=[0.34, 0.44, 0.22]
            ),
            "TechSupport": rng.choice(
                ["Yes", "No", "No internet service"], n_rows, p=[0.29, 0.49, 0.22]
            ),
            "StreamingTV": rng.choice(
                ["Yes", "No", "No internet service"], n_rows, p=[0.38, 0.40, 0.22]
            ),
            "StreamingMovies": rng.choice(
                ["Yes", "No", "No internet service"], n_rows, p=[0.39, 0.39, 0.22]
            ),
            "Contract": rng.choice(
                ["Month-to-month", "One year", "Two year"], n_rows, p=[0.55, 0.21, 0.24]
            ),
            "PaperlessBilling": rng.choice(["Yes", "No"], n_rows, p=[0.59, 0.41]),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                 "Credit card (automatic)"],
                n_rows,
                p=[0.34, 0.23, 0.22, 0.21],
            ),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
    )

    # Churn driven by a latent risk score so the signal is learnable but noisy.
    risk = (
        -0.045 * (tenure - 30)
        + 0.018 * (monthly - 65)
        + 1.15 * (df["Contract"] == "Month-to-month")
        + 0.55 * (df["InternetService"] == "Fiber optic")
        - 0.40 * (df["TechSupport"] == "Yes")
        - 0.35 * (df["OnlineSecurity"] == "Yes")
        + 0.30 * (df["PaymentMethod"] == "Electronic check")
        + 0.25 * (df["PaperlessBilling"] == "Yes")
        + 0.20 * df["SeniorCitizen"]
        - 0.25 * (df["Partner"] == "Yes")
        - 0.15 * (df["Dependents"] == "Yes")
    )
    prob = 1.0 / (1.0 + np.exp(-(risk - 0.55)))
    df["Churn"] = np.where(rng.random(n_rows) < prob, "Yes", "No")
    return df


def load_data(path: str | None) -> pd.DataFrame:
    """Load the Telco CSV when available, else fall back to synthetic data."""
    if path:
        csv = Path(path).expanduser()
        if not csv.exists():
            raise FileNotFoundError(f"CSV not found: {csv}")
        print(f"[data] loading {csv}")
        df = pd.read_csv(csv)
    else:
        default = Path(__file__).resolve().parent / "telco_churn.csv"
        if default.exists():
            print(f"[data] loading {default}")
            df = pd.read_csv(default)
        else:
            print("[data] no CSV found - generating 3000 synthetic rows")
            df = generate_synthetic()
    return df


# --------------------------------------------------------------------------
# 2. Cleaning
# --------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ID/leakage columns, coerce dtypes, impute missing values."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for col in ("customerID", "CustomerID"):
        if col in df.columns:
            df = df.drop(columns=col)

    # TotalCharges is blank-string in the real dataset for new customers.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Median for numeric, most frequent for categorical.
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            n = int(df[col].isna().sum())
            df[col] = df[col].fillna(df[col].median())
            print(f"[clean] {col}: imputed {n} missing value(s) with median")

    for col in df.select_dtypes(include=["object", "string"]).columns:
        if df[col].isna().any():
            n = int(df[col].isna().sum())
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
            print(f"[clean] {col}: imputed {n} missing value(s) with mode")

    before = len(df)
    df = df.dropna(subset=["Churn"]) if "Churn" in df.columns else df
    if len(df) != before:
        print(f"[clean] dropped {before - len(df)} row(s) without a Churn label")

    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[clean] final shape: {df.shape}")
    return df


# --------------------------------------------------------------------------
# 3. Split + preprocessing
# --------------------------------------------------------------------------
def build_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    """One-hot encode categoricals, impute + scale numerics."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )


# --------------------------------------------------------------------------
# 4. Reusable build entry point (used by main() and by app.py)
# --------------------------------------------------------------------------
def build_and_save(
    data_path: str | None = None,
    output_dir: Path | str = MODELS_DIR,
    test_size: float = 0.2,
    verbose: bool = True,
):
    """Clean -> split -> fit -> evaluate -> save.

    Returns a 3-tuple ``(model, scaler, metadata)``:

    * ``model``    - the fitted :class:`~sklearn.pipeline.Pipeline`
                     (ColumnTransformer + XGBClassifier)
    * ``scaler``   - the fitted numeric imputer/scaler pipeline
    * ``metadata`` - dict with feature names, options, ranges and ROC-AUC

    Kept side-effect free on import so app.py can call it to rebuild the
    artifacts in-process when a saved pickle cannot be unpickled.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    df = clean(load_data(data_path))

    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' is missing from the data.")

    y = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    X = df.drop(columns=["Churn"])

    cat_cols = [c for c in CATEGORICAL if c in X.columns]
    num_cols = [c for c in NUMERIC if c in X.columns]
    leftover = [c for c in X.columns if c not in cat_cols + num_cols]
    if leftover:
        log(f"[warn] ignoring unexpected columns: {leftover}")
        X = X.drop(columns=leftover)

    log(f"[data] {X.shape[0]} rows | {len(num_cols)} numeric, {len(cat_cols)} categorical")
    log(f"[data] churn rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(cat_cols, num_cols)

    # scale_pos_weight keeps the minority (churn) class from being ignored.
    spw = float((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=5,
                    min_child_weight=2,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    scale_pos_weight=spw,
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    log("\n[train] fitting XGBoost...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    log("\n" + "=" * 62)
    log("CLASSIFICATION REPORT (hold-out test set)")
    log("=" * 62)
    log(
        classification_report(
            y_test, y_pred, target_names=["No Churn", "Churn"], digits=4
        )
    )
    log("Confusion matrix [[TN FP] [FN TP]]:")
    log(str(confusion_matrix(y_test, y_pred)))
    log(f"\nROC-AUC: {auc:.4f}")
    log("=" * 62)

    # Feature names after the ColumnTransformer, for the Streamlit app.
    feature_names = list(model.named_steps["preprocessor"].get_feature_names_out())

    # A standalone scaler object, saved as the brief asks.
    scaler = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    scaler.fit(X_train[num_cols])

    metadata = {
        "feature_names": feature_names,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "roc_auc": round(float(auc), 4),
        "options": {c: sorted(df[c].astype(str).unique().tolist()) for c in cat_cols},
        "ranges": {
            c: {
                "min": float(df[c].min()),
                "max": float(df[c].max()),
                "median": float(df[c].median()),
            }
            for c in num_cols
        },
    }

    joblib.dump(model, output_dir / "churn_model.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    with open(output_dir / "feature_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    log(f"\n[save] model      -> {output_dir / 'churn_model.joblib'}")
    log(f"[save] scaler     -> {output_dir / 'scaler.joblib'}")
    log(f"[save] metadata   -> {output_dir / 'feature_metadata.json'}")

    return model, scaler, metadata


# --------------------------------------------------------------------------
# 5. CLI
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train the churn model.")
    parser.add_argument("--data", default=None, help="Path to a Telco churn CSV.")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    model, scaler, metadata = build_and_save(
        data_path=args.data, test_size=args.test_size
    )
    return model, scaler, metadata


if __name__ == "__main__":
    main()
