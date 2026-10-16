"""
app.py - Streamlit front end for the churn model trained by train.py

The model is never read from disk: train.build_and_save() is called on every
app run, so the pipeline is always built live in memory with the versions
installed here. The call also refreshes the files in models/, which keeps the
saved artifacts in step with the running app.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"

# train.py must be importable no matter which directory streamlit was started from.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import train  # noqa: E402  (needs HERE on sys.path first)

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

FALLBACK_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}

FALLBACK_RANGES = {
    "tenure": {"min": 1.0, "max": 72.0, "median": 29.0},
    "MonthlyCharges": {"min": 18.0, "max": 120.0, "median": 70.0},
    "TotalCharges": {"min": 18.0, "max": 8600.0, "median": 1400.0},
}


@st.cache_resource(show_spinner=False)
def train_model():
    """Train the pipeline once, in memory, via train.build_and_save().

    The @st.cache_resource decorator above caches the return value for the life
    of the server process, so the 400-tree fit runs on the first page load only.
    Every later rerun, widget interaction and browser session gets the cached
    pipeline back instead of retraining.

    Nothing is unpickled: the model is built from the raw data with the versions
    installed here, so a stale or incompatible artifact on disk can never break
    the dashboard. The call still refreshes models/, keeping those files current.

    Returns (model, scaler, metadata, notice). ``notice`` describes the outcome
    of the training run, or why it could not be completed.
    """
    if not hasattr(train, "build_and_save"):
        return None, None, {}, {
            "status": "failed",
            "reason": "train.py does not expose build_and_save().",
            "detail": "Update train.py to the version that returns "
                      "(model, scaler, metadata).",
        }

    try:
        result = train.build_and_save(data_path=None, output_dir=MODELS_DIR)
    except Exception as exc:
        return None, None, {}, {
            "status": "failed",
            "reason": f"train.build_and_save() raised {type(exc).__name__}.",
            "detail": traceback.format_exc(limit=5),
        }

    # build_and_save returns (model, scaler, metadata). Older revisions returned
    # only (model, metadata); accept both shapes rather than failing here.
    model = scaler = None
    meta: dict = {}
    if isinstance(result, tuple):
        if len(result) == 3:
            model, scaler, meta = result
        elif len(result) == 2:
            model, meta = result
        elif len(result) == 1:
            model = result[0]
    else:
        model = result

    if model is None:
        return None, None, {}, {
            "status": "failed",
            "reason": "train.build_and_save() returned no model.",
            "detail": f"Got: {type(result).__name__}",
        }

    return model, scaler, meta or {}, {
        "status": "trained",
        "reason": "Model trained live in memory from train.build_and_save().",
        "detail": "",
    }


def risk_band(p: float) -> tuple[str, str]:
    if p >= 0.65:
        return "High risk", "#d62728"
    if p >= 0.35:
        return "Medium risk", "#ff7f0e"
    return "Low risk", "#2ca02c"


def per_customer_contributions(clf, row_matrix: np.ndarray, n_features: int) -> np.ndarray:
    """SHAP-style contributions for one row, via the XGBoost booster."""
    booster = clf.get_booster()
    dm = xgb.DMatrix(row_matrix, missing=np.nan)
    contribs = np.asarray(booster.predict(dm, pred_contribs=True))
    flat = contribs[0]
    # XGBoost appends a final bias term; drop it to line up with feature names.
    return flat[:n_features]


# ------------------------------------------------------------ sidebar ---
# train_model() is @st.cache_resource'd: the first load fits the model, every
# later rerun and session reuses that same in-memory pipeline.
with st.spinner("Training the churn model..."):
    model, scaler, meta, load_notice = train_model()

st.title("📉 Customer Churn Predictor")
st.caption("XGBoost model trained live by `train.py` on the Telco Churn schema.")

if load_notice and load_notice["status"] == "trained":
    st.success(
        "Model trained **live in memory** this run via `train.build_and_save()` "
        f"— nothing was read from a saved pickle. Artifacts refreshed in "
        f"`{MODELS_DIR.name}/`."
    )

if model is None:
    st.error("Could not train the model.")
    st.markdown(
        f"**Why it failed:** {load_notice['reason'] if load_notice else 'unknown'}"
    )
    if load_notice and load_notice.get("detail"):
        with st.expander("Error details"):
            st.code(load_notice["detail"], language="text")
    st.stop()

# The scaler is not used for prediction (the model pipeline scales internally),
# but it is returned by build_and_save() and kept so the artifact set is complete.
if scaler is not None:
    st.sidebar.caption(
        f"Numeric scaler built ({len(meta.get('numeric_features', []))} features)."
    )

options = meta.get("options", FALLBACK_OPTIONS)
ranges = meta.get("ranges", FALLBACK_RANGES)
auc = meta.get("roc_auc")

if auc:
    st.caption(f"Hold-out ROC-AUC of this training run: **{auc:.4f}**")

with st.sidebar:
    st.header("Customer details")

    st.subheader("Account")
    tenure = st.slider(
        "Tenure (months)", 0, int(ranges.get("tenure", {}).get("max", 72)), 12
    )
    contract = st.selectbox("Contract type", options.get("Contract", FALLBACK_OPTIONS["Contract"]))
    payment = st.selectbox(
        "Payment method", options.get("PaymentMethod", FALLBACK_OPTIONS["PaymentMethod"])
    )
    paperless = st.radio(
        "Paperless billing",
        options.get("PaperlessBilling", FALLBACK_OPTIONS["PaperlessBilling"]),
        horizontal=True,
    )

    st.subheader("Charges")
    monthly = st.slider(
        "Monthly charges ($)", 0.0, 200.0, float(ranges.get("MonthlyCharges", {}).get("median", 70.0)), 0.5
    )
    auto_total = monthly * max(tenure, 1)
    total = st.number_input(
        "Total charges ($)", 0.0, 20000.0, float(round(auto_total, 2)), 10.0
    )

    st.subheader("Services")
    internet = st.selectbox(
        "Internet service", options.get("InternetService", FALLBACK_OPTIONS["InternetService"])
    )
    phone = st.selectbox("Phone service", options.get("PhoneService", FALLBACK_OPTIONS["PhoneService"]))
    multiple = st.selectbox(
        "Multiple lines", options.get("MultipleLines", FALLBACK_OPTIONS["MultipleLines"])
    )
    security = st.selectbox(
        "Online security", options.get("OnlineSecurity", FALLBACK_OPTIONS["OnlineSecurity"])
    )
    backup = st.selectbox(
        "Online backup", options.get("OnlineBackup", FALLBACK_OPTIONS["OnlineBackup"])
    )
    protection = st.selectbox(
        "Device protection", options.get("DeviceProtection", FALLBACK_OPTIONS["DeviceProtection"])
    )
    support = st.selectbox("Tech support", options.get("TechSupport", FALLBACK_OPTIONS["TechSupport"]))
    tv = st.selectbox("Streaming TV", options.get("StreamingTV", FALLBACK_OPTIONS["StreamingTV"]))
    movies = st.selectbox(
        "Streaming movies", options.get("StreamingMovies", FALLBACK_OPTIONS["StreamingMovies"])
    )

    st.subheader("Profile")
    gender = st.radio("Gender", options.get("gender", FALLBACK_OPTIONS["gender"]), horizontal=True)
    senior = st.checkbox("Senior citizen")
    partner = st.radio("Partner", options.get("Partner", FALLBACK_OPTIONS["Partner"]), horizontal=True)
    dependents = st.radio(
        "Dependents", options.get("Dependents", FALLBACK_OPTIONS["Dependents"]), horizontal=True
    )

customer = {
    "gender": gender,
    "SeniorCitizen": int(senior),
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    "OnlineSecurity": security,
    "OnlineBackup": backup,
    "DeviceProtection": protection,
    "TechSupport": support,
    "StreamingTV": tv,
    "StreamingMovies": movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
}

# ------------------------------------------------------------- prediction ---
row = pd.DataFrame([customer])
row = row.reindex(columns=list(model.feature_names_in_), fill_value=np.nan)

proba = float(model.predict_proba(row)[0, 1])
label, color = risk_band(proba)

left, right = st.columns([1, 1.4])

with left:
    st.subheader("Churn probability")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%", "font": {"size": 44}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 35], "color": "#e8f5e9"},
                    {"range": [35, 65], "color": "#fff3e0"},
                    {"range": [65, 100], "color": "#ffebee"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
            },
        )
    )
    fig.update_layout(height=280, margin=dict(t=20, b=10, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"### :{'red' if color == '#d62728' else 'orange' if color == '#ff7f0e' else 'green'}[{label}]")

    if proba >= 0.65:
        st.warning("Prioritise retention: offer a contract upgrade or a service credit.")
    elif proba >= 0.35:
        st.info("Worth a check-in call — look at contract length and support tickets.")
    else:
        st.success("Likely to stay. No action needed.")

with right:
    st.subheader("What is driving this prediction")
    try:
        prep = model.named_steps["preprocessor"]
        clf = model.named_steps["classifier"]
        names = list(prep.get_feature_names_out())
        row_t = prep.transform(row)
        row_t = np.asarray(row_t.toarray() if hasattr(row_t, "toarray") else row_t)[0]
        contrib = per_customer_contributions(clf, np.asarray([row_t]), len(names))
        shap = pd.DataFrame({"feature": names, "impact": contrib})
        shap["abs"] = shap["impact"].abs()
        top = shap.sort_values("abs", ascending=False).head(12).sort_values("impact")
        top["feature"] = top["feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)

        fig2 = px.bar(
            top,
            x="impact",
            y="feature",
            orientation="h",
            color="impact",
            color_continuous_scale=["#2ca02c", "#f0f0f0", "#d62728"],
        )
        fig2.update_layout(
            height=380,
            coloraxis_showscale=False,
            margin=dict(t=20, b=10, l=10, r=10),
            xaxis_title="pushes churn ↑  /  pushes retention ↓",
            yaxis_title="",
        )
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as exc:  # pragma: no cover - display fallback
        st.info(f"Per-customer explanation unavailable ({exc}). Showing global importance.")

    st.divider()
    st.subheader("Global feature importance")
    clf = model.named_steps["classifier"]
    names = list(model.named_steps["preprocessor"].get_feature_names_out())
    imp = pd.DataFrame({"feature": names, "importance": clf.feature_importances_})
    imp["feature"] = imp["feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
    imp = imp.sort_values("importance", ascending=False).head(15).sort_values("importance")

    fig3 = px.bar(imp, x="importance", y="feature", orientation="h", color_discrete_sequence=["#1f77b4"])
    fig3.update_layout(height=420, margin=dict(t=20, b=10, l=10, r=10), yaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.subheader("Customer snapshot")
snap = pd.DataFrame(
    {"field": list(customer.keys()), "value": [str(v) for v in customer.values()]}
)
st.dataframe(snap, use_container_width=True, hide_index=True)

st.caption("Built with XGBoost + Streamlit. Predictions are model estimates, not decisions.")
