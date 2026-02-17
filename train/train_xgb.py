import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# ==============================
# Load dataset
# ==============================
DATA_PATH = "Traffic_preprocessed_EV_with_queue_augmented.csv"
df = pd.read_csv(DATA_PATH)
print("Dataset loaded:", df.shape)

# ==============================
# Define target + leakage columns
# ==============================
TARGET = "Clearance_Distance_km_final"

leakage_cols = [
    "Clearance_Distance_km_final",
    "Clearance_Time_min_final",
    "Clearance_Distance_km",
    "Clearance_Distance_km_noisy",
    "Clearance_Time_min",
    "Clearance_Time_min_noisy"
]

# Remove only existing columns
leakage_cols = [c for c in leakage_cols if c in df.columns]

# Features (X) and target (y)
X = df.drop(columns=leakage_cols)
y = df[TARGET]

# ==============================
# One-hot encoding
# ==============================
print("\nPerforming one-hot encoding...")
X = pd.get_dummies(X, drop_first=True)

# Save feature columns for inference
feature_cols = X.columns.tolist()
joblib.dump(feature_cols, "xgb_feature_columns.joblib")

print(f"Saved feature columns: xgb_feature_columns.joblib")

# ==============================
# Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("Train:", X_train.shape, "Test:", X_test.shape)

# ==============================
# XGBoost Model
# ==============================
xgb_model = xgb.XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

print("\nTraining XGBoost model...")
xgb_model.fit(X_train, y_train)

# ==============================
# Predictions
# ==============================
train_pred = xgb_model.predict(X_train)
test_pred = xgb_model.predict(X_test)

# ==============================
# Metrics
# ==============================
train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print("\n--- XGBoost Evaluation ---")
print(f"Train MAE : {train_mae:.6f} km")
print(f"Test MAE : {test_mae:.6f} km")
print(f"Train RMSE: {train_rmse:.6f} km")
print(f"Test RMSE : {test_rmse:.6f} km")
print(f"Train R² : {train_r2:.6f}")
print(f"Test R² : {test_r2:.6f}")

# ==============================
# Save model
# ==============================
joblib.dump(xgb_model, "xgb_clearance_model.joblib")

print("\nModel saved as: xgb_clearance_model.joblib")
print("Feature columns saved as: xgb_feature_columns.joblib")
