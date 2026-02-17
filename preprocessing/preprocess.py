"""
This module is included for compatibility with the ML model loader.
It is not used during Streamlit inference.

In your original project, this file:
- Loaded the CSV dataset
- One-hot encoded features
- Saved xgb_feature_columns.joblib
- Performed train-test split

In the Streamlit simulation, we DO NOT retrain models.
We only need this file present so imports do not fail.
"""

import pandas as pd
import joblib
from pathlib import Path

# Base directory reference
BASE_DIR = Path(__file__).resolve().parent.parent

def preprocess_dataset():
    """
    Dummy placeholder used for compatibility.
    Prevents crashes if some module imports it.
    """
    raise RuntimeError(
        "preprocess_dataset() is not used inside Streamlit simulation. "
        "ML models must be pre-trained and placed inside the models/ folder."
    )
