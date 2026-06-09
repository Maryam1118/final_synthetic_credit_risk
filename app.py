import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
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

warnings.filterwarnings("ignore")

TARGET = "SeriousDlqin2yrs"
ID_COLUMNS = {"Unnamed: 0", "Id"}
DEFAULT_TRAIN_PATH = Path("data/cs-training.csv")
DEFAULT_TEST_PATH = Path("data/cs-test.csv")


def load_credit_data(path_or_file) -> pd.DataFrame:
    df = pd.read_csv(path_or_file)
    df = df.drop(columns=[col for col in df.columns if col in ID_COLUMNS], errors="ignore")
    return df


def clean_credit_data(df: pd.DataFrame, has_target: bool = True) -> pd.DataFrame:
    df = df.copy()

    if has_target and TARGET in df.columns:
        df = df.dropna(subset=[TARGET])
        df[TARGET] = df[TARGET].astype(int)

    numeric_columns = [col for col in df.columns if col != TARGET]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "age" in df.columns:
        df = df[df["age"].fillna(0) > 0]

    return df.reset_index(drop=True)


def build_preprocessor(feature_columns):
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_columns,
            )
        ],
        remainder="drop",
    )


def build_classifier(feature_columns, random_state=42):
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(feature_columns)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_ctgan(real_train, epochs=80, batch_size=500, random_state=42):
    try:
        from sdv.metadata import SingleTableMetadata
        from sdv.sampling import Condition
        from sdv.single_table import CTGANSynthesizer
    except ImportError as exc:
        raise RuntimeError(
            "The SDV package is required for CTGAN. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_train)
    metadata.update_column(TARGET, sdtype="categorical")

    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=epochs,
        batch_size=batch_size,
        verbose=False,
        cuda=False,
    )
    synthesizer.fit(real_train)
    return synthesizer, Condition({TARGET: 1}, num_rows=1)


def generate_synthetic_minority(
    synthesizer,
    minority_count,
    majority_count,
    target_ratio=0.35,
):
    from sdv.sampling import Condition

    desired_minority = int((target_ratio * majority_count) / max(1 - target_ratio, 1e-6))
    rows_to_generate = max(0, desired_minority - minority_count)

    if rows_to_generate == 0:
        return pd.DataFrame()

    condition = Condition({TARGET: 1}, num_rows=rows_to_generate)
    synthetic = synthesizer.sample_from_conditions([condition])
    synthetic[TARGET] = 1
    return synthetic


def train_pipeline(
    df,
    synthetic_ratio=0.35,
    ctgan_epochs=80,
    sample_limit=None,
    random_state=42,
):
    df = clean_credit_data(df)

    if sample_limit and len(df) > sample_limit:
        df = df.sample(sample_limit, random_state=random_state).reset_index(drop=True)

    feature_columns = [col for col in df.columns if col != TARGET]
    train_df, valid_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df[TARGET],
        random_state=random_state,
    )

    counts = train_df[TARGET].value_counts()
    minority_count = int(counts.get(1, 0))
    majority_count = int(counts.get(0, 0))

    synthesizer, _ = train_ctgan(
        train_df,
        epochs=ctgan_epochs,
        random_state=random_state,
    )
    synthetic_df = generate_synthetic_minority(
        synthesizer,
        minority_count=minority_count,
        majority_count=majority_count,
        target_ratio=synthetic_ratio,
    )

    augmented_train = pd.concat([train_df, synthetic_df], ignore_index=True)
    model = build_classifier(feature_columns, random_state=random_state)
    model.fit(augmented_train[feature_columns], augmented_train[TARGET])

    probabilities = model.predict_proba(valid_df[feature_columns])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(valid_df[TARGET], probabilities),
        "average_precision": average_precision_score(valid_df[TARGET], probabilities),
        "accuracy": accuracy_score(valid_df[TARGET], predictions),
        "precision": precision_score(valid_df[TARGET], predictions, zero_division=0),
        "recall": recall_score(valid_df[TARGET], predictions, zero_division=0),
        "f1": f1_score(valid_df[TARGET], predictions, zero_division=0),
        "synthetic_rows": len(synthetic_df),
        "training_rows": len(train_df),
        "augmented_rows": len(augmented_train),
    }

    return {
        "model": model,
        "synthesizer": synthesizer,
        "feature_columns": feature_columns,
        "train_df": train_df,
        "valid_df": valid_df,
        "synthetic_df": synthetic_df,
        "augmented_train": augmented_train,
        "probabilities": probabilities,
        "predictions": predictions,
        "metrics": metrics,
    }


def predict_test_file(model, feature_columns, test_df):
    clean_test = clean_credit_data(test_df.drop(columns=[TARGET], errors="ignore"), has_target=False)
    probabilities = model.predict_proba(clean_test[feature_columns])[:, 1]
    return pd.DataFrame(
        {
            "Id": np.arange(1, len(clean_test) + 1),
            "Probability": probabilities,
        }
    )


def render_streamlit_app():
    st.set_page_config(
        page_title="CTGAN Credit Risk AI",
        page_icon="CT",
        layout="wide",
    )

    st.title("Synthetic Financial Data Generation for Credit Risk Assessment")
    st.caption(
        "CTGAN-based synthetic minority generation plus a credit default prediction model "
        "for the Give Me Some Credit dataset."
    )

    with st.sidebar:
        st.header("Data")
        uploaded_train = st.file_uploader("Training CSV", type=["csv"])
        uploaded_test = st.file_uploader("Optional test CSV", type=["csv"])

        st.header("CTGAN")
        sample_limit = st.slider(
            "Training sample size",
            min_value=5000,
            max_value=150000,
            value=30000,
            step=5000,
            help="Lower values train faster. Use 150000 for the full dataset.",
        )
        ctgan_epochs = st.slider("CTGAN epochs", 10, 300, 80, 10)
        synthetic_ratio = st.slider(
            "Target default share after augmentation",
            0.10,
            0.50,
            0.35,
            0.05,
        )
        train_button = st.button("Train CTGAN + Credit Model", type="primary")

    train_source = uploaded_train if uploaded_train is not None else DEFAULT_TRAIN_PATH

    if not Path(DEFAULT_TRAIN_PATH).exists() and uploaded_train is None:
        st.warning("Upload `cs-training.csv` or place it at `data/cs-training.csv`.")
        return

    df = load_credit_data(train_source)
    df = clean_credit_data(df)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", f"{len(df):,}")
    metric_cols[1].metric("Features", len([c for c in df.columns if c != TARGET]))
    metric_cols[2].metric("Default Rate", f"{df[TARGET].mean():.2%}")
    metric_cols[3].metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(25), use_container_width=True)

    st.subheader("Target Distribution")
    st.bar_chart(df[TARGET].value_counts().sort_index())

    if train_button:
        with st.spinner("Training CTGAN and the credit-risk model. This can take a few minutes."):
            results = train_pipeline(
                df,
                synthetic_ratio=synthetic_ratio,
                ctgan_epochs=ctgan_epochs,
                sample_limit=sample_limit,
            )

        st.success("Training complete.")

        st.subheader("Model Metrics")
        metrics = results["metrics"]
        cols = st.columns(6)
        cols[0].metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        cols[1].metric("Avg Precision", f"{metrics['average_precision']:.3f}")
        cols[2].metric("Recall", f"{metrics['recall']:.3f}")
        cols[3].metric("Precision", f"{metrics['precision']:.3f}")
        cols[4].metric("F1", f"{metrics['f1']:.3f}")
        cols[5].metric("Synthetic Rows", f"{metrics['synthetic_rows']:,}")

        cm = confusion_matrix(results["valid_df"][TARGET], results["predictions"])
        st.write("Confusion Matrix")
        st.dataframe(
            pd.DataFrame(
                cm,
                index=["Actual 0", "Actual 1"],
                columns=["Predicted 0", "Predicted 1"],
            ),
            use_container_width=True,
        )

        with st.expander("Classification report"):
            report = classification_report(
                results["valid_df"][TARGET],
                results["predictions"],
                zero_division=0,
            )
            st.code(report)

        st.subheader("Synthetic Default Examples")
        if results["synthetic_df"].empty:
            st.info("No synthetic rows were needed for the selected target ratio.")
        else:
            st.dataframe(results["synthetic_df"].head(25), use_container_width=True)
            st.download_button(
                "Download synthetic defaults",
                data=results["synthetic_df"].to_csv(index=False),
                file_name="synthetic_ctgan_defaults.csv",
                mime="text/csv",
            )

        test_source = uploaded_test if uploaded_test is not None else (
            DEFAULT_TEST_PATH if DEFAULT_TEST_PATH.exists() else None
        )
        if test_source is not None:
            test_df = pd.read_csv(test_source)
            submission = predict_test_file(
                results["model"],
                results["feature_columns"],
                test_df,
            )
            st.subheader("Kaggle-style Submission")
            st.dataframe(submission.head(25), use_container_width=True)
            st.download_button(
                "Download predictions",
                data=submission.to_csv(index=False),
                file_name="ctgan_credit_risk_submission.csv",
                mime="text/csv",
            )


def run_cli(args):
    df = load_credit_data(args.train)
    results = train_pipeline(
        df,
        synthetic_ratio=args.synthetic_ratio,
        ctgan_epochs=args.epochs,
        sample_limit=args.sample_limit,
        random_state=args.random_state,
    )
    print("CTGAN credit-risk training complete")
    for key, value in results["metrics"].items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    if args.test:
        test_df = pd.read_csv(args.test)
        submission = predict_test_file(
            results["model"],
            results["feature_columns"],
            test_df,
        )
        submission.to_csv(args.output, index=False)
        print(f"Saved predictions to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CTGAN credit-risk model.")
    parser.add_argument("--cli", action="store_true", help="Run without Streamlit UI.")
    parser.add_argument("--train", default=str(DEFAULT_TRAIN_PATH), help="Training CSV path.")
    parser.add_argument("--test", default=str(DEFAULT_TEST_PATH), help="Optional test CSV path.")
    parser.add_argument("--output", default="ctgan_credit_risk_submission.csv")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--sample-limit", type=int, default=30000)
    parser.add_argument("--synthetic-ratio", type=float, default=0.35)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cli:
        run_cli(args)
    else:
        render_streamlit_app()
