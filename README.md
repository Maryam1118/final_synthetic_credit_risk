# CTGAN Credit Risk Prediction

Synthetic financial data generation and default-risk prediction for the **Give Me Some Credit** dataset.

This project trains a CTGAN model to generate synthetic high-risk credit profiles, augments the real training data, and trains a credit-risk classifier that predicts the probability of serious delinquency within two years.

## Project Files

```text
.
├── app.py
├── requirements.txt
├── README.md
└── data/
    ├── cs-training.csv
    ├── cs-test.csv
    └── sampleEntry.csv
```

## Features

- Loads the Give Me Some Credit training and test CSV files.
- Cleans missing values and removes row IDs.
- Trains a CTGAN synthesizer with SDV.
- Generates conditional synthetic records for `SeriousDlqin2yrs = 1`.
- Trains a Random Forest credit-risk classifier on real plus synthetic data.
- Shows ROC AUC, average precision, recall, precision, F1, and confusion matrix.
- Creates a Kaggle-style prediction CSV with `Id` and `Probability`.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

## Command Line Run

```bash
python app.py --cli --train data/cs-training.csv --test data/cs-test.csv --output ctgan_credit_risk_submission.csv
```

For a faster demo:

```bash
python app.py --cli --sample-limit 10000 --epochs 20
```

## Workflow

1. Load and clean the credit dataset.
2. Split real data into train and validation data.
3. Train CTGAN on the real training records.
4. Generate synthetic default-risk records.
5. Train a credit-risk classifier on real plus synthetic data.
6. Evaluate on real validation data.
7. Export test-set default probabilities.

## Note

This is a research and learning project. It should not be used for real lending decisions without deeper validation, fairness testing, calibration, monitoring, and compliance approval.
