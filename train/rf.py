import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# ==============================
# Load dataset
# ==============================
DATA_PATH = "Traffic_preprocessed_EV_with_queue_augmented.csv"
df = pd.read_csv(DATA_PATH)
print("Dataset loaded:", df.shape)
# ==============================
# Define target + remove leakage columns
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
# Remove only the cols that exist
leakage_cols = [c for c in leakage_cols if c in df.columns]
X = df.drop(columns=leakage_cols)
y = df[TARGET]
# ==============================
# One-hot encoding
# ==============================
X = pd.get_dummies(X, drop_first=True)
# Save feature columns for inference
feature_cols = X.columns.tolist()
joblib.dump(feature_cols, "rf_feature_columns.joblib")
# ==============================
# Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.20, random_state=42
)
print("Train:", X_train.shape, "Test:", X_test.shape)
# ==============================
# Random Forest Model
# ==============================
rf_model = RandomForestRegressor(
n_estimators=600,
max_depth=None,
min_samples_split=2,
min_samples_leaf=1,
random_state=42,
n_jobs=-1
)
print("\nTraining Random Forest model...")
rf_model.fit(X_train, y_train)
# ==============================
# Predictions
# ==============================
train_pred = rf_model.predict(X_train)
test_pred = rf_model.predict(X_test)
# ==============================
# Metrics
# ==============================
train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)
print("\n--- Random Forest Evaluation ---")
print(f"Train MAE : {train_mae:.6f} km")
print(f"Test MAE : {test_mae:.6f} km")
print(f"Train RMSE: {train_rmse:.6f} km")
print(f"Test RMSE : {test_rmse:.6f} km")
print(f"Train R² : {train_r2:.6f}")
print(f"Test R² : {test_r2:.6f}")
# ==============================
# Save model
# ==============================
joblib.dump(rf_model, "rf_clearance_model.joblib")
print("\nModel saved as: rf_clearance_model.joblib")
print("Feature columns saved as: rf_feature_columns.joblib")