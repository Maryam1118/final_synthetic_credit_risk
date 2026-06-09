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


st.set_page_config(page_title="CTGAN Credit Risk AI", layout="wide")


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
    from sdv.metadata import SingleTableMetadata
    from sdv.sampling import Condition
    from sdv.single_table import CTGANSynthesizer

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

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_df)
    metadata.update_column(TARGET, sdtype="categorical")

    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=ctgan_epochs,
        verbose=False,
    )
    synthesizer.fit(train_df)

    counts = train_df[TARGET].value_counts()
    minority_count = int(counts.get(1, 0))
    majority_count = int(counts.get(0, 0))

    desired_minority = int((synthetic_ratio * majority_count) / max(1 - synthetic_ratio, 1e-6))
    rows_to_generate = max(0, desired_minority - minority_count)

    if rows_to_generate > 0:
        condition = Condition({TARGET: 1}, num_rows=rows_to_generate)
        synthetic_df = synthesizer.sample_from_conditions([condition])
        synthetic_df[TARGET] = 1
    else:
        synthetic_df = pd.DataFrame(columns=train_df.columns)

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
                    n_estimators=200,
                    min_samples_leaf=20,
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

    return model, features, synthetic_df, valid_df, predictions, metrics, matrix, report


def make_submission(model, features, test_data):
    test_data = clean_data(test_data.drop(columns=[TARGET], errors="ignore"), has_target=False)
    probabilities = model.predict_proba(test_data[features])[:, 1]

    return pd.DataFrame(
        {
            "Id": np.arange(1, len(test_data) + 1),
            "Probability": probabilities,
        }
    )


st.title("Synthetic Financial Data Generation for Credit Risk Assessment")
st.caption("CTGAN-based synthetic data generation and credit default prediction.")

with st.sidebar:
    st.header("Upload Data")
    uploaded_train = st.file_uploader("Upload cs-training.csv", type=["csv"])
    uploaded_test = st.file_uploader("Optional cs-test.csv", type=["csv"])

    st.header("Training Settings")
    sample_limit = st.slider("Training sample size", 5000, 150000, 20000, 5000)
    ctgan_epochs = st.slider("CTGAN epochs", 5, 100, 20, 5)
    synthetic_ratio = st.slider("Target default share", 0.10, 0.50, 0.30, 0.05)

    train_button = st.button("Train CTGAN + Model", type="primary")

train_source = uploaded_train

if train_source is None and DEFAULT_TRAIN_PATH.exists():
    train_source = DEFAULT_TRAIN_PATH

if train_source is None:
    st.info("Upload `cs-training.csv` from the sidebar to start.")
    st.stop()

try:
    df = clean_data(load_credit_data(train_source))
except Exception as error:
    st.error("Could not load the training data.")
    st.exception(error)
    st.stop()

cols = st.columns(4)
cols[0].metric("Rows", f"{len(df):,}")
cols[1].metric("Features", len([c for c in df.columns if c != TARGET]))
cols[2].metric("Default Rate", f"{df[TARGET].mean():.2%}")
cols[3].metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

st.subheader("Dataset Preview")
st.dataframe(df.head(25), use_container_width=True)

st.subheader("Target Distribution")
st.bar_chart(df[TARGET].value_counts().sort_index())

if not train_button:
    st.stop()

with st.spinner("Training CTGAN and credit risk model. Please wait..."):
    try:
        (
            model,
            features,
            synthetic_df,
            valid_df,
            predictions,
            metrics,
            matrix,
            report,
        ) = train_credit_model(df, sample_limit, ctgan_epochs, synthetic_ratio)
    except Exception as error:
        st.error("Training failed. Full error:")
        st.exception(error)
        st.stop()

st.success("Training complete.")

st.subheader("Model Metrics")
metric_cols = st.columns(6)
metric_cols[0].metric("ROC AUC", f"{metrics['ROC AUC']:.3f}")
metric_cols[1].metric("Avg Precision", f"{metrics['Average Precision']:.3f}")
metric_cols[2].metric("Accuracy", f"{metrics['Accuracy']:.3f}")
metric_cols[3].metric("Precision", f"{metrics['Precision']:.3f}")
metric_cols[4].metric("Recall", f"{metrics['Recall']:.3f}")
metric_cols[5].metric("F1", f"{metrics['F1']:.3f}")

st.metric("Synthetic Rows Generated", f"{metrics['Synthetic Rows']:,}")

st.subheader("Confusion Matrix")
st.dataframe(
    pd.DataFrame(
        matrix,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    ),
    use_container_width=True,
)

with st.expander("Classification Report"):
    st.code(report)

st.subheader("Synthetic Default Examples")
if synthetic_df.empty:
    st.info("No synthetic rows were needed for this target ratio.")
else:
    st.dataframe(synthetic_df.head(25), use_container_width=True)
    st.download_button(
        "Download Synthetic Data",
        synthetic_df.to_csv(index=False),
        "synthetic_ctgan_defaults.csv",
        "text/csv",
    )

test_source = uploaded_test

if test_source is None and DEFAULT_TEST_PATH.exists():
    test_source = DEFAULT_TEST_PATH

if test_source is not None:
    try:
        test_df = pd.read_csv(test_source)
        submission = make_submission(model, features, test_df)

        st.subheader("Prediction Submission")
        st.dataframe(submission.head(25), use_container_width=True)

        st.download_button(
            "Download Predictions",
            submission.to_csv(index=False),
            "ctgan_credit_risk_submission.csv",
            "text/csv",
        )
    except Exception as error:
        st.error("Could not create test predictions.")
        st.exception(error)
