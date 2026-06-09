from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

TARGET = "SeriousDlqin2yrs"
ID_COLUMNS = ["Unnamed: 0", "Id"]
DEFAULT_TRAIN_PATH = Path("data/cs-training.csv")
DEFAULT_TEST_PATH = Path("data/cs-test.csv")

st.set_page_config(page_title="Evoastra CTGAN Credit Risk", layout="wide")


def inject_dark_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 5%, rgba(255, 95, 87, 0.20), transparent 30%),
                radial-gradient(circle at 90% 5%, rgba(37, 224, 196, 0.16), transparent 28%),
                linear-gradient(135deg, #08070d 0%, #121021 55%, #1a1024 100%);
            color: #f8f4ff;
        }

        section[data-testid="stSidebar"] {
            background: rgba(10, 8, 17, 0.96);
            border-right: 1px solid rgba(255, 255, 255, 0.12);
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1240px;
        }

        h1, h2, h3, h4, p, label {
            color: #f8f4ff !important;
        }

        .hero {
            padding: 30px 32px;
            border: 1px solid rgba(255, 255, 255, 0.13);
            border-radius: 22px;
            background:
                linear-gradient(135deg, rgba(255, 95, 87, 0.18), rgba(37, 224, 196, 0.10)),
                rgba(20, 16, 31, 0.78);
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
            margin-bottom: 26px;
        }

        .eyebrow {
            display: inline-flex;
            color: #25e0c4;
            border: 1px solid rgba(37, 224, 196, 0.35);
            background: rgba(37, 224, 196, 0.08);
            padding: 7px 11px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .hero h1 {
            font-size: clamp(2.1rem, 5vw, 4.5rem);
            line-height: 1.02;
            margin: 18px 0 14px;
        }

        .hero p {
            max-width: 850px;
            color: #c8bed8 !important;
            font-size: 1.05rem;
            line-height: 1.75;
            margin: 0;
        }

        .pill-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 22px;
        }

        .pill {
            border: 1px solid rgba(255, 255, 255, 0.13);
            background: rgba(255, 255, 255, 0.07);
            color: #f8f4ff;
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 700;
        }

        div[data-testid="metric-container"] {
            background: rgba(20, 16, 31, 0.82);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.20);
        }

        div[data-testid="metric-container"] label {
            color: #c8bed8 !important;
        }

        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: #ffffff !important;
        }

        .panel {
            background: rgba(20, 16, 31, 0.76);
            border: 1px solid rgba(255, 255, 255, 0.11);
            border-radius: 18px;
            padding: 20px;
            margin: 12px 0 24px;
        }

        .status-ok {
            color: #25e0c4 !important;
            font-weight: 800;
        }

        .stButton > button,
        .stDownloadButton > button {
            background: linear-gradient(135deg, #ff5f57, #9d63ff) !important;
            color: white !important;
            border: 0 !important;
            border-radius: 14px !important;
            padding: 0.75rem 1rem !important;
            font-weight: 800 !important;
            box-shadow: 0 16px 34px rgba(157, 99, 255, 0.22);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            filter: brightness(1.08);
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 16px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Evoastra AI Research Program</div>
            <h1>CTGAN Credit Risk Intelligence Studio</h1>
            <p>
                Generate synthetic credit profiles, rebalance default-risk data,
                train a credit risk model, and export prediction files from one polished dashboard.
            </p>
            <div class="pill-row">
                <span class="pill">CTGAN Synthetic Data</span>
                <span class="pill">Default Risk Prediction</span>
                <span class="pill">Streamlit Cloud Ready</span>
                <span class="pill">Sample Data Loader</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_credit_data(file_or_path):
    data = pd.read_csv(file_or_path)
    data = data.drop(columns=[c for c in ID_COLUMNS if c in data.columns], errors="ignore")
    return data


def clean_data(data, has_target=True):
    data = data.copy()

    if has_target:
        if TARGET not in data.columns:
            raise ValueError(f"Training CSV must contain `{TARGET}` column.")
        data = data.dropna(subset=[TARGET])
        data[TARGET] = data[TARGET].astype(int)

    for col in data.columns:
        if col != TARGET:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "age" in data.columns:
        data = data[data["age"].fillna(0) > 0]

    return data.reset_index(drop=True)


def fallback_synthetic_data(train_df, rows_to_generate):
    minority_df = train_df[train_df[TARGET] == 1]

    if rows_to_generate <= 0 or minority_df.empty:
        return pd.DataFrame(columns=train_df.columns), "No synthetic rows needed"

    synthetic_df = minority_df.sample(
        rows_to_generate,
        replace=True,
        random_state=42,
    ).reset_index(drop=True)

    numeric_cols = [c for c in synthetic_df.columns if c != TARGET]
    rng = np.random.default_rng(42)

    for col in numeric_cols:
        synthetic_df[col] = pd.to_numeric(synthetic_df[col], errors="coerce")
        col_std = synthetic_df[col].std()

        if pd.isna(col_std) or col_std == 0:
            col_std = 0.01

        noise = rng.normal(0, col_std * 0.01, len(synthetic_df))
        synthetic_df[col] = synthetic_df[col] + noise

    synthetic_df[TARGET] = 1
    return synthetic_df, "Fallback synthetic oversampling used because CTGAN could not run on this server"


def train_credit_model(data, sample_limit, ctgan_epochs, synthetic_ratio):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    data = clean_data(data)

    if sample_limit and len(data) > sample_limit:
        data = data.sample(sample_limit, random_state=42).reset_index(drop=True)

    features = [c for c in data.columns if c != TARGET]

    train_df, valid_df = train_test_split(
        data,
        test_size=0.2,
        stratify=data[TARGET],
        random_state=42,
    )

    counts = train_df[TARGET].value_counts()
    minority_count = int(counts.get(1, 0))
    majority_count = int(counts.get(0, 0))

    desired_minority = int((synthetic_ratio * majority_count) / max(1 - synthetic_ratio, 1e-6))
    rows_to_generate = max(0, desired_minority - minority_count)
    rows_to_generate = min(rows_to_generate, 200)

    synthetic_status = "CTGAN synthetic data generated successfully"

    try:
        from sdv.metadata import SingleTableMetadata
        from sdv.sampling import Condition
        from sdv.single_table import CTGANSynthesizer

        small_ctgan_train = train_df.sample(
            min(len(train_df), 500),
            random_state=42,
        ).reset_index(drop=True)

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(small_ctgan_train)
        metadata.update_column(TARGET, sdtype="categorical")

        synthesizer = CTGANSynthesizer(
            metadata,
            epochs=ctgan_epochs,
            batch_size=50,
            verbose=False,
            cuda=False,
        )

        synthesizer.fit(small_ctgan_train)

        if rows_to_generate > 0:
            condition = Condition({TARGET: 1}, num_rows=rows_to_generate)
            synthetic_df = synthesizer.sample_from_conditions([condition])
            synthetic_df[TARGET] = 1
        else:
            synthetic_df = pd.DataFrame(columns=train_df.columns)

    except Exception:
        synthetic_df, synthetic_status = fallback_synthetic_data(train_df, rows_to_generate)

    for col in train_df.columns:
        if col not in synthetic_df.columns:
            synthetic_df[col] = np.nan

    synthetic_df = synthetic_df[train_df.columns]
    augmented_train = pd.concat([train_df, synthetic_df], ignore_index=True)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                features,
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    min_samples_leaf=10,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(augmented_train[features], augmented_train[TARGET])

    probabilities = model.predict_proba(valid_df[features])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "ROC AUC": roc_auc_score(valid_df[TARGET], probabilities),
        "Average Precision": average_precision_score(valid_df[TARGET], probabilities),
        "Accuracy": accuracy_score(valid_df[TARGET], predictions),
        "Precision": precision_score(valid_df[TARGET], predictions, zero_division=0),
        "Recall": recall_score(valid_df[TARGET], predictions, zero_division=0),
        "F1": f1_score(valid_df[TARGET], predictions, zero_division=0),
        "Synthetic Rows": len(synthetic_df),
    }

    matrix = confusion_matrix(valid_df[TARGET], predictions)
    report = classification_report(valid_df[TARGET], predictions, zero_division=0)

    return model, features, synthetic_df, metrics, matrix, report, synthetic_status


def make_submission(model, features, test_data):
    test_data = clean_data(test_data.drop(columns=[TARGET], errors="ignore"), has_target=False)

    for col in features:
        if col not in test_data.columns:
            test_data[col] = np.nan

    test_data = test_data[features]
    probabilities = model.predict_proba(test_data)[:, 1]

    return pd.DataFrame(
        {
            "Id": np.arange(1, len(test_data) + 1),
            "Probability": probabilities,
        }
    )


inject_dark_theme()
render_hero()

if "use_sample_data" not in st.session_state:
    st.session_state.use_sample_data = False

with st.sidebar:
    st.markdown("## EVOASTRA")
    st.caption("Synthetic credit-risk AI lab")
    st.divider()

    if st.button("Load Sample Data", use_container_width=True):
        st.session_state.use_sample_data = True

    uploaded_train = st.file_uploader("Upload training CSV", type=["csv"])
    uploaded_test = st.file_uploader("Optional test CSV", type=["csv"])

    st.divider()
    st.markdown("### Training Settings")

    sample_limit = st.slider("Training sample size", 200, 2000, 500, 100)
    ctgan_epochs = st.slider("CTGAN epochs", 1, 10, 1, 1)
    synthetic_ratio = st.slider("Target default share", 0.10, 0.30, 0.15, 0.05)

    train_button = st.button("Train CTGAN + Model", type="primary", use_container_width=True)

    st.divider()
    st.caption("Streamlit Cloud works best with small CTGAN settings.")

train_source = uploaded_train

if train_source is None and st.session_state.use_sample_data:
    if DEFAULT_TRAIN_PATH.exists():
        train_source = DEFAULT_TRAIN_PATH
    else:
        st.warning("Sample data was not found. Upload `cs-training.csv` to continue.")

if train_source is None and DEFAULT_TRAIN_PATH.exists():
    st.info("Click **Load Sample Data** in the sidebar, or upload your own training CSV.")
    st.stop()

if train_source is None:
    st.info("Upload `cs-training.csv` in the sidebar to start.")
    st.stop()

try:
    df = clean_data(load_credit_data(train_source))
except Exception as error:
    st.error("Could not load the training data.")
    st.exception(error)
    st.stop()

st.markdown("## Portfolio Overview")

cols = st.columns(4)
cols[0].metric("Credit Records", f"{len(df):,}")
cols[1].metric("Model Features", len([c for c in df.columns if c != TARGET]))
cols[2].metric("Observed Default Rate", f"{df[TARGET].mean():.2%}")
cols[3].metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

left, right = st.columns([1.1, 1])

with left:
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True, height=360)

with right:
    st.markdown("### Target Distribution")

    distribution = df[TARGET].value_counts().sort_index()
    chart_df = pd.DataFrame(
        {
            "Risk Class": ["No Serious Delinquency", "Serious Delinquency"],
            "Records": [int(distribution.get(0, 0)), int(distribution.get(1, 0))],
        }
    )

    st.bar_chart(chart_df, x="Risk Class", y="Records", color="#25e0c4")

if not train_button:
    st.markdown(
        """
        <div class="panel">
            <h3>Ready to Train</h3>
            <p style="color:#c8bed8;">
                Use the sidebar controls to tune sample size, CTGAN epochs, and target default share.
                Click <b>Train CTGAN + Model</b> when you are ready.
            </p>
            <p class="status-ok">Cloud-safe defaults are already selected.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

with st.spinner("Training CTGAN and credit risk model. Please wait..."):
    try:
        (
            model,
            features,
            synthetic_df,
            metrics,
            matrix,
            report,
            synthetic_status,
        ) = train_credit_model(df, sample_limit, ctgan_epochs, synthetic_ratio)
    except Exception as error:
        st.error("Training failed. Full error:")
        st.exception(error)
        st.stop()

st.success("Training complete.")
st.info(synthetic_status)

st.markdown("## Executive Metrics")

metric_cols = st.columns(6)
metric_cols[0].metric("ROC AUC", f"{metrics['ROC AUC']:.3f}")
metric_cols[1].metric("Avg Precision", f"{metrics['Average Precision']:.3f}")
metric_cols[2].metric("Accuracy", f"{metrics['Accuracy']:.3f}")
metric_cols[3].metric("Precision", f"{metrics['Precision']:.3f}")
metric_cols[4].metric("Recall", f"{metrics['Recall']:.3f}")
metric_cols[5].metric("F1", f"{metrics['F1']:.3f}")

st.metric("Synthetic Rows Generated", f"{metrics['Synthetic Rows']:,}")

matrix_col, synthetic_col = st.columns([0.9, 1.1])

with matrix_col:
    st.markdown("### Confusion Matrix")
    st.dataframe(
        pd.DataFrame(
            matrix,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        ),
        use_container_width=True,
    )

with synthetic_col:
    st.markdown("### Synthetic Default Examples")

    if synthetic_df.empty:
        st.info("No synthetic rows were generated.")
    else:
        st.dataframe(synthetic_df.head(12), use_container_width=True, height=300)
        st.download_button(
            "Download Synthetic Data",
            synthetic_df.to_csv(index=False),
            "synthetic_ctgan_defaults.csv",
            "text/csv",
            use_container_width=True,
        )

with st.expander("Classification Report"):
    st.code(report)

test_source = uploaded_test

if test_source is None and st.session_state.use_sample_data and DEFAULT_TEST_PATH.exists():
    test_source = DEFAULT_TEST_PATH

if test_source is not None:
    try:
        test_df = pd.read_csv(test_source)
        submission = make_submission(model, features, test_df)

        st.markdown("## Prediction Submission")
        st.dataframe(submission.head(25), use_container_width=True)

        st.download_button(
            "Download Predictions",
            submission.to_csv(index=False),
            "ctgan_credit_risk_submission.csv",
            "text/csv",
            use_container_width=True,
        )

    except Exception as error:
        st.error("Could not create test predictions.")
        st.exception(error)
