# Customer Churn Prediction — ML Pipeline

An end-to-end machine learning pipeline that predicts customer churn for a telecom
provider, built on the Telco Customer Churn data structure. It trains an XGBoost
classifier and serves real-time predictions, per-customer explanations and feature
importance through an interactive Streamlit app.

---

## Overview

Customer churn is one of the most valuable things a subscription business can
predict. This project covers the full lifecycle — data preparation, model training,
evaluation, persistence and an interactive front end — in two clean, self-contained
scripts.

**What it does**

| Stage | Detail |
| --- | --- |
| **Data** | Loads a Telco churn CSV if present, otherwise generates 3,000 synthetic rows with the identical schema (15 categorical, 4 numeric features) |
| **Cleaning** | Drops ID columns, coerces `TotalCharges` from blank strings to numeric, imputes missing values (median / mode), removes duplicates and unlabelled rows |
| **Encoding** | One-hot encoding for categoricals, median imputation + standard scaling for numerics — all inside a scikit-learn `ColumnTransformer` pipeline |
| **Model** | `XGBClassifier` (400 estimators, learning rate 0.05, depth 5, `scale_pos_weight` for class imbalance) |
| **Evaluation** | Classification report, confusion matrix and ROC-AUC on a stratified hold-out set |
| **Serving** | Streamlit app with sidebar inputs, a live probability gauge, SHAP-style per-customer contributions and global feature importance |
| **Persistence** | Model, scaler and feature metadata saved with `joblib` into `models/` |

**Latest run (synthetic data, 3,000 rows):** ROC-AUC **0.7483**, accuracy **0.68**
on a 600-row hold-out set.

---

## Tech stack

- **XGBoost** — gradient-boosted trees for the classifier
- **scikit-learn** — preprocessing pipelines, train/test split, metrics
- **Pandas / NumPy** — data wrangling, cleaning and synthetic data generation
- **Streamlit** — interactive web front end
- **Plotly** — gauge, contribution and importance charts
- **joblib** — model and artifact serialization

---

## Project structure

```
ml_pipeline_project/
├── train.py                  # cleaning, training, evaluation, artifact saving
├── app.py                    # Streamlit prediction app
├── requirements.txt          # dependencies
├── README.md
└── models/                   # created by train.py
    ├── churn_model.joblib
    ├── scaler.joblib
    └── feature_metadata.json
```

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/ml_pipeline_project.git
cd ml_pipeline_project
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Train the model

```bash
python train.py
```

To train on a real Telco churn CSV instead of synthetic data:

```bash
python train.py --data path/to/telco_churn.csv
```

Alternatively, place a file named `telco_churn.csv` in the project root and `train.py`
will pick it up automatically. The script prints the classification report, confusion
matrix and ROC-AUC, then writes the model artifacts into `models/`.

### Step 2 — Launch the app

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser. Adjust tenure, charges, contract
type and service options in the sidebar to see the churn probability update in real
time, along with the factors driving each individual prediction.

> **Note:** the app requires `models/churn_model.joblib` — run `train.py` first.

---

## Model inputs

| Type | Features |
| --- | --- |
| **Numeric** | `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen` |
| **Categorical** | `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` |

**Target:** `Churn` (`Yes` / `No`)

---

## License

Released under the MIT License. Predictions are model estimates and should support,
not replace, retention decisions.
