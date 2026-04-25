"""
Customer Churn Prediction – Streamlit App
==========================================
Notebook source: churn_prediction.ipynb
Model         : Logistic Regression (scikit-learn)
Preprocessing : StandardScaler + OneHotEncoder via make_column_transformer

Run
---
    streamlit run app.py

Expected files in the same directory
-------------------------------------
    log_reg.pkl       – trained LogisticRegression model
    transformer.pkl   – fitted ColumnTransformer (scaler + OHE)

If neither file exists the app will train a fresh model from
WA_Fn-UseC_-Telco-Customer-Churn.csv (also in the same directory).
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import io
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page config & global style
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {"primary": "#6C63FF", "danger": "#EF476F", "success": "#06D6A0", "bg": "#F8F9FA"}

st.markdown(
    f"""
    <style>
        .main-header {{
            font-size: 2.4rem; font-weight: 800;
            color: {PALETTE['primary']}; margin-bottom: 0;
        }}
        .sub-header {{ color: #888; font-size: 1rem; margin-top: 0; }}
        .metric-card {{
            background: white; border-radius: 12px;
            padding: 1.2rem 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,.08);
            text-align: center;
        }}
        .metric-label {{ font-size: .85rem; color: #888; margin-bottom: .2rem; }}
        .metric-value {{ font-size: 2rem; font-weight: 700; color: {PALETTE['primary']}; }}
        .churn-yes {{
            background: {PALETTE['danger']}22; border-left: 5px solid {PALETTE['danger']};
            border-radius: 8px; padding: 1rem 1.5rem;
        }}
        .churn-no {{
            background: {PALETTE['success']}22; border-left: 5px solid {PALETTE['success']};
            border-radius: 8px; padding: 1rem 1.5rem;
        }}
        .section-title {{
            font-size: 1.3rem; font-weight: 700;
            color: #333; border-bottom: 2px solid {PALETTE['primary']}33;
            padding-bottom: .4rem; margin-top: 1.5rem;
        }}
        [data-testid="stSidebar"] {{background: #1E1E2E; color: white;}}
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSlider label {{color: #ccc !important;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Feature schema (mirrors notebook)
# ─────────────────────────────────────────────────────────────────────────────
CATEGORICAL = [
    "gender", "seniorcitizen", "partner", "dependents",
    "phoneservice", "multiplelines", "internetservice",
    "onlinesecurity", "onlinebackup", "deviceprotection",
    "techsupport", "streamingtv", "streamingmovies",
    "contract", "paperlessbilling", "paymentmethod",
]
NUMERICAL = ["tenure", "monthlycharges", "totalcharges"]

CAT_OPTIONS = {
    "gender":          ["Male", "Female"],
    "seniorcitizen":   ["0", "1"],
    "partner":         ["Yes", "No"],
    "dependents":      ["Yes", "No"],
    "phoneservice":    ["Yes", "No"],
    "multiplelines":   ["Yes", "No", "No phone service"],
    "internetservice": ["DSL", "Fiber optic", "No"],
    "onlinesecurity":  ["Yes", "No", "No internet service"],
    "onlinebackup":    ["Yes", "No", "No internet service"],
    "deviceprotection":["Yes", "No", "No internet service"],
    "techsupport":     ["Yes", "No", "No internet service"],
    "streamingtv":     ["Yes", "No", "No internet service"],
    "streamingmovies": ["Yes", "No", "No internet service"],
    "contract":        ["Month-to-month", "One year", "Two year"],
    "paperlessbilling":["Yes", "No"],
    "paymentmethod":   [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Utility: load / train model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_or_train():
    """
    Tries to load pre-saved model + transformer.
    Falls back to training from the CSV if pkl files are absent.
    Returns (model, transformer, x_test, y_test, feature_names).
    """
    try:
        model       = pickle.load(open("log_reg.pkl",     "rb"))
        transformer = pickle.load(open("transformer.pkl", "rb"))
        # Rebuild a small test set for metric display
        df = _load_and_clean_df()
        _, df_test = train_test_split(df, test_size=0.2, random_state=1)
        x_test = transformer.transform(df_test[CATEGORICAL + NUMERICAL])
        y_test = df_test["churn"]
        feature_names = transformer.get_feature_names_out()
        return model, transformer, x_test, y_test, feature_names
    except FileNotFoundError:
        pass

    # ── Train from scratch ──────────────────────────────────────────────────
    df = _load_and_clean_df()
    df_full_train, df_test   = train_test_split(df, test_size=0.2, random_state=1)
    df_train,      df_valid  = train_test_split(df_full_train, test_size=0.2, random_state=1)

    ohe        = OneHotEncoder(drop="first", handle_unknown="ignore")
    scaler     = StandardScaler()
    transformer = make_column_transformer(
        (scaler, NUMERICAL),
        (ohe,    CATEGORICAL),
        remainder="drop",
        verbose_feature_names_out=False,
    )

    x_train = transformer.fit_transform(df_train[CATEGORICAL + NUMERICAL])
    x_test  = transformer.transform(df_test[CATEGORICAL + NUMERICAL])
    y_train, y_test = df_train["churn"], df_test["churn"]

    model = LogisticRegression(solver="liblinear", random_state=1)
    model.fit(x_train, y_train)

    pickle.dump(model,       open("log_reg.pkl",     "wb"))
    pickle.dump(transformer, open("transformer.pkl", "wb"))

    feature_names = transformer.get_feature_names_out()
    return model, transformer, x_test, y_test, feature_names


def _load_and_clean_df():
    """Load and clean the Telco CSV (mirrors notebook preprocessing)."""
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df.drop("customerID", axis=1, inplace=True, errors="ignore")
    df.columns = df.columns.str.lower()
    df["totalcharges"]  = pd.to_numeric(df["totalcharges"], errors="coerce")
    df["totalcharges"].fillna(df["totalcharges"].mean(), inplace=True)
    df["seniorcitizen"] = df["seniorcitizen"].astype("object")
    df["churn"]         = (df["churn"] == "Yes").astype(int)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – user input
# ─────────────────────────────────────────────────────────────────────────────
def build_sidebar() -> dict:
    """Render sidebar widgets and return a dict of user inputs."""
    st.sidebar.markdown(
        "<h2 style='color:white;'>🔧 Customer Profile</h2>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    inputs = {}

    st.sidebar.markdown("**📋 Demographics**")
    inputs["gender"]        = st.sidebar.selectbox("Gender",         CAT_OPTIONS["gender"])
    inputs["seniorcitizen"] = st.sidebar.selectbox("Senior Citizen", CAT_OPTIONS["seniorcitizen"])
    inputs["partner"]       = st.sidebar.selectbox("Has Partner",    CAT_OPTIONS["partner"])
    inputs["dependents"]    = st.sidebar.selectbox("Has Dependents", CAT_OPTIONS["dependents"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📱 Services**")
    inputs["phoneservice"]    = st.sidebar.selectbox("Phone Service",     CAT_OPTIONS["phoneservice"])
    inputs["multiplelines"]   = st.sidebar.selectbox("Multiple Lines",    CAT_OPTIONS["multiplelines"])
    inputs["internetservice"] = st.sidebar.selectbox("Internet Service",  CAT_OPTIONS["internetservice"])
    inputs["onlinesecurity"]  = st.sidebar.selectbox("Online Security",   CAT_OPTIONS["onlinesecurity"])
    inputs["onlinebackup"]    = st.sidebar.selectbox("Online Backup",     CAT_OPTIONS["onlinebackup"])
    inputs["deviceprotection"]= st.sidebar.selectbox("Device Protection", CAT_OPTIONS["deviceprotection"])
    inputs["techsupport"]     = st.sidebar.selectbox("Tech Support",      CAT_OPTIONS["techsupport"])
    inputs["streamingtv"]     = st.sidebar.selectbox("Streaming TV",      CAT_OPTIONS["streamingtv"])
    inputs["streamingmovies"] = st.sidebar.selectbox("Streaming Movies",  CAT_OPTIONS["streamingmovies"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**💳 Billing**")
    inputs["contract"]         = st.sidebar.selectbox("Contract Type",    CAT_OPTIONS["contract"])
    inputs["paperlessbilling"] = st.sidebar.selectbox("Paperless Billing",CAT_OPTIONS["paperlessbilling"])
    inputs["paymentmethod"]    = st.sidebar.selectbox("Payment Method",   CAT_OPTIONS["paymentmethod"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Account**")
    inputs["tenure"]         = st.sidebar.slider("Tenure (months)",        0, 72,  12)
    inputs["monthlycharges"] = st.sidebar.slider("Monthly Charges ($)",    0.0, 120.0, 65.0, step=0.5)
    inputs["totalcharges"]   = st.sidebar.number_input(
        "Total Charges ($)", min_value=0.0, max_value=10000.0,
        value=float(inputs["tenure"] * inputs["monthlycharges"]),
        step=10.0,
    )

    return inputs

# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────
def predict_single(model, transformer, inputs: dict):
    """Run one-row prediction; return (label, probability)."""
    row = pd.DataFrame([inputs])
    row.columns = row.columns.str.lower()
    X = transformer.transform(row[CATEGORICAL + NUMERICAL])
    proba = model.predict_proba(X)[0][1]
    label = "Churn" if proba >= 0.5 else "No Churn"
    return label, proba

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(model, x_test, y_test):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm   = confusion_matrix(y_test, model.predict(x_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_churn_distribution(df_raw):
    counts = df_raw["churn"].value_counts()
    labels = ["No Churn", "Churn"]
    colors = [PALETTE["success"], PALETTE["danger"]]
    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(11); at.set_fontweight("bold")
    ax.set_title("Churn Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=15):
    coef = model.coef_[0]
    idx  = np.argsort(np.abs(coef))[-top_n:]
    names, vals = np.array(feature_names)[idx], coef[idx]
    colors = [PALETTE["danger"] if v > 0 else PALETTE["success"] for v in vals]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(names, vals, color=colors)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("Coefficient (log-odds)", fontsize=10)
    ax.set_title(f"Top {top_n} Feature Coefficients", fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig


def plot_prob_gauge(proba: float):
    """Simple horizontal bar gauge for churn probability."""
    fig, ax = plt.subplots(figsize=(5, 1.2))
    ax.barh(0, 1,  color="#E0E0E0", height=0.5)
    color = PALETTE["danger"] if proba >= 0.5 else PALETTE["success"]
    ax.barh(0, proba, color=color, height=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    ax.text(proba + 0.02, 0, f"{proba:.1%}", va="center", fontsize=13, fontweight="bold", color=color)
    ax.set_title("Churn Probability", fontsize=11, pad=4)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df_raw):
    num_df = df_raw[NUMERICAL + ["churn"]].copy()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        num_df.corr(), annot=True, fmt=".2f",
        cmap="coolwarm", ax=ax, linewidths=.5,
        annot_kws={"size": 10},
    )
    ax.set_title("Numerical Correlation Heatmap", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Batch prediction helpers
# ─────────────────────────────────────────────────────────────────────────────
def run_batch(model, transformer, df_upload: pd.DataFrame) -> pd.DataFrame:
    """Apply model to an uploaded CSV; returns df with Prediction + Probability cols."""
    df = df_upload.copy()
    df.columns = df.columns.str.lower()

    # Drop customerID if present
    df.drop("customerid", axis=1, inplace=True, errors="ignore")

    # Coerce totalcharges
    if "totalcharges" in df.columns:
        df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
        df["totalcharges"].fillna(df["totalcharges"].mean(), inplace=True)

    if "seniorcitizen" in df.columns:
        df["seniorcitizen"] = df["seniorcitizen"].astype("object")

    missing = [c for c in CATEGORICAL + NUMERICAL if c not in df.columns]
    if missing:
        st.error(f"Uploaded CSV is missing columns: {missing}")
        return pd.DataFrame()

    X      = transformer.transform(df[CATEGORICAL + NUMERICAL])
    probas = model.predict_proba(X)[:, 1]
    df["Probability"] = probas.round(4)
    df["Prediction"]  = ["Churn" if p >= 0.5 else "No Churn" for p in probas]
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        "<p class='main-header'>📉 Customer Churn Predictor</p>"
        "<p class='sub-header'>Telco dataset · Logistic Regression · scikit-learn</p>",
        unsafe_allow_html=True,
    )

    # ── Load model ───────────────────────────────────────────────────────────
    try:
        model, transformer, x_test, y_test, feature_names = load_or_train()
    except Exception as e:
        st.error(
            f"**Could not load or train the model.**\n\n"
            f"Make sure `WA_Fn-UseC_-Telco-Customer-Churn.csv` is in the same folder.\n\n"
            f"Error: {e}"
        )
        st.stop()

    # ── Sidebar inputs ───────────────────────────────────────────────────────
    inputs = build_sidebar()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_pred, tab_metrics, tab_viz, tab_batch = st.tabs(
        ["🔮 Prediction", "📊 Model Metrics", "📈 Visualizations", "📂 Batch Prediction"]
    )

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 – Single Prediction
    # ═════════════════════════════════════════════════════════════════════════
    with tab_pred:
        st.markdown("<p class='section-title'>Single Customer Prediction</p>", unsafe_allow_html=True)
        st.markdown("Adjust the customer profile in the **sidebar**, then click **Predict**.")

        col_btn, _ = st.columns([1, 3])
        predict_clicked = col_btn.button("🔮 Predict Churn", use_container_width=True, type="primary")

        if predict_clicked:
            label, proba = predict_single(model, transformer, inputs)

            # Result card
            css_class = "churn-yes" if label == "Churn" else "churn-no"
            icon       = "🚨" if label == "Churn" else "✅"
            st.markdown(
                f"<div class='{css_class}'>"
                f"<h2>{icon} Prediction: <b>{label}</b></h2>"
                f"<p style='margin:0;font-size:1.1rem;'>Confidence: <b>{proba:.1%}</b></p>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.pyplot(plot_prob_gauge(proba))

            st.markdown("---")
            st.markdown("**📋 Input Summary**")
            summary = pd.DataFrame(inputs.items(), columns=["Feature", "Value"])
            st.dataframe(summary, use_container_width=True, hide_index=True)

        else:
            st.info("👈  Set customer attributes in the sidebar and click **Predict Churn**.")

        # ── Sample data hint ─────────────────────────────────────────────────
        with st.expander("📌 Sample Input Values"):
            sample = {
                "gender": "Male", "seniorcitizen": "0", "partner": "No",
                "dependents": "No", "tenure": 2, "phoneservice": "Yes",
                "multiplelines": "No", "internetservice": "Fiber optic",
                "onlinesecurity": "No", "onlinebackup": "No",
                "deviceprotection": "No", "techsupport": "No",
                "streamingtv": "No", "streamingmovies": "No",
                "contract": "Month-to-month", "paperlessbilling": "Yes",
                "paymentmethod": "Electronic check",
                "monthlycharges": 70.35, "totalcharges": 140.70,
            }
            st.json(sample)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 – Model Metrics
    # ═════════════════════════════════════════════════════════════════════════
    with tab_metrics:
        st.markdown("<p class='section-title'>Model Performance on Test Set</p>", unsafe_allow_html=True)

        y_pred      = model.predict(x_test)
        y_proba_pos = model.predict_proba(x_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        roc  = roc_auc_score(y_test, y_proba_pos)

        m1, m2, m3, m4, m5 = st.columns(5)
        for col, label, val in zip(
            [m1, m2, m3, m4, m5],
            ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            [acc, prec, rec, f1, roc],
        ):
            col.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value'>{val:.2%}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        col_cm, col_rep = st.columns([1, 1])

        with col_cm:
            st.markdown("**Confusion Matrix**")
            st.pyplot(plot_confusion_matrix(model, x_test, y_test))

        with col_rep:
            st.markdown("**Classification Report**")
            report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
            st.code(report, language="text")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3 – Visualizations
    # ═════════════════════════════════════════════════════════════════════════
    with tab_viz:
        st.markdown("<p class='section-title'>Data & Feature Visualizations</p>", unsafe_allow_html=True)

        try:
            df_raw = _load_and_clean_df()

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Churn Distribution**")
                st.pyplot(plot_churn_distribution(df_raw))
            with col_b:
                st.markdown("**Numerical Correlation Heatmap**")
                st.pyplot(plot_correlation_heatmap(df_raw))

            st.markdown("---")
            st.markdown("**Feature Importance (Model Coefficients)**")
            st.pyplot(plot_feature_importance(model, feature_names, top_n=15))

            st.markdown(
                "_Red bars → positive log-odds (increase churn risk)_ &nbsp;|&nbsp; "
                "_Green bars → negative log-odds (reduce churn risk)_"
            )

            # Tenure vs Monthly Charges scatter
            st.markdown("---")
            st.markdown("**Tenure vs Monthly Charges by Churn**")
            fig, ax = plt.subplots(figsize=(7, 4))
            colors_map = {0: PALETTE["success"], 1: PALETTE["danger"]}
            for churn_val, group in df_raw.groupby("churn"):
                ax.scatter(
                    group["tenure"], group["monthlycharges"],
                    c=colors_map[churn_val], alpha=0.3, s=10,
                    label="Churn" if churn_val else "No Churn",
                )
            ax.set_xlabel("Tenure (months)")
            ax.set_ylabel("Monthly Charges ($)")
            ax.legend()
            ax.set_title("Tenure vs Monthly Charges", fontweight="bold")
            fig.tight_layout()
            st.pyplot(fig)

        except FileNotFoundError:
            st.warning("CSV file not found – visualizations require `WA_Fn-UseC_-Telco-Customer-Churn.csv`.")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 4 – Batch Prediction
    # ═════════════════════════════════════════════════════════════════════════
    with tab_batch:
        st.markdown("<p class='section-title'>Batch Prediction via CSV Upload</p>", unsafe_allow_html=True)
        st.markdown(
            "Upload a CSV with the same columns as the Telco dataset "
            "(without the `Churn` column). The app will add `Prediction` and `Probability` columns."
        )

        uploaded = st.file_uploader("📂 Upload CSV", type=["csv"])
        if uploaded:
            df_upload = pd.read_csv(uploaded)
            st.markdown(f"**Preview** ({len(df_upload):,} rows)")
            st.dataframe(df_upload.head(), use_container_width=True)

            if st.button("⚡ Run Batch Prediction", type="primary"):
                result_df = run_batch(model, transformer, df_upload)
                if not result_df.empty:
                    n_churn = (result_df["Prediction"] == "Churn").sum()
                    st.success(
                        f"Done! **{n_churn}** predicted churners out of **{len(result_df)}** customers "
                        f"({n_churn/len(result_df):.1%})."
                    )
                    st.dataframe(result_df[["Prediction", "Probability"]].join(
                        df_upload.reset_index(drop=True)), use_container_width=True)

                    # Download button
                    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Results as CSV",
                        data=csv_bytes,
                        file_name="churn_predictions.csv",
                        mime="text/csv",
                    )

        st.markdown("---")
        with st.expander("📋 Required CSV Columns"):
            st.markdown(
                "Your file must contain **all** of the following columns "
                "(case-insensitive, `customerID` is optional and will be ignored):"
            )
            cols_df = pd.DataFrame(
                {"Categorical": CATEGORICAL + [""] * (len(NUMERICAL) - len(CATEGORICAL)),
                 "Numerical":   NUMERICAL + [""] * (len(CATEGORICAL) - len(NUMERICAL))}
            )
            st.dataframe(
                pd.DataFrame({"Column": CATEGORICAL + NUMERICAL}),
                use_container_width=True, hide_index=True,
            )

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#aaa;font-size:.85rem;'>"
        "Customer Churn Predictor · Logistic Regression · Telco Dataset"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
