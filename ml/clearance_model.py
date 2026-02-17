import joblib
import pandas as pd
from preprocessing.feature_loader import align_features
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# Model file paths
MODEL_PATHS = {
    "XGBoost": BASE / "models" / "xgb_clearance_model.joblib",
    "Tuned XGBoost": BASE / "models" / "xgb_tuned_model.joblib",
    "Random Forest": BASE / "models" / "rf_clearance_model.joblib",
}

# Feature column files per model
FEATURE_PATHS = {
    "XGBoost": BASE / "models" / "xgb_feature_columns.joblib",
    "Tuned XGBoost": BASE / "models" / "xgb_feature_columns.joblib",
    "Random Forest": BASE / "models" / "rf_feature_columns.joblib",
}


def load_model(model_name):
    """Load the correct model and its corresponding feature list."""
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Invalid model name: {model_name}")

    model = joblib.load(MODEL_PATHS[model_name])

    feature_file = FEATURE_PATHS.get(model_name)
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file missing for {model_name}: {feature_file}")

    feature_cols = joblib.load(feature_file)

    return model, feature_cols


def predict_clearance(sample_dict, model_name="XGBoost"):
    """Predicts clearance distance using any ML model."""
    model, feature_cols = load_model(model_name)

    df = pd.DataFrame([sample_dict])
    df = pd.get_dummies(df, drop_first=True)

    # FIX: Create missing columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # FIX: Remove extra columns
    df = df[feature_cols]

    pred = model.predict(df)[0]
    return float(pred)
