import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_COLS_FILE = BASE_DIR / "models" / "xgb_feature_columns.joblib"


def load_feature_columns():
    """Load saved training feature columns."""
    return joblib.load(FEATURE_COLS_FILE)


def align_features(df):
    """
    Ensures a DataFrame matches the exact feature schema used during training.
    Missing columns → added as zeros.
    Extra columns → dropped.
    Columns → ordered identically.
    """

    feature_cols = load_feature_columns()

    # Add missing columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Drop unknown columns
    df = df[feature_cols]

    return df
