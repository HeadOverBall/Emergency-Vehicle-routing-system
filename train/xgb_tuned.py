# xgb_optuna_tune_and_train.py
# Requires: optuna, xgboost, scikit-learn, pandas, joblib, numpy
# Install: pip install optuna xgboost scikit-learn pandas joblib
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, ma
import xgboost as xgb
import optuna
import warnings
warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_PATH = "Traffic_preprocessed_EV_with_queue_augmented.csv"
# -------------------------
# Load & prepare data
# -------------------------
df = pd.read_csv(DATA_PATH)
print("Dataset loaded:", df.shape)
TARGET = "Clearance_Distance_km_final"
# drop leakage columns and target only
# drop leakage columns and target only
drop_cols = [
    TARGET,
    "Clearance_Time_min_final",
    "Clearance_Distance_km",
    "Clearance_Distance_km_noisy",
    "Clearance_Time_min",
    "Clearance_Time_min_noisy",
    "Clearance_Distance_Final_km",
    "Clearance_Time_Final_min",
]

drop_cols = [c for c in drop_cols if c in df.columns]
X = df.drop(columns=drop_cols)
y = df[TARGET]
# one-hot encode
X = pd.get_dummies(X, drop_first=True)
# Save feature columns (for inference)
feature_cols = X.columns.tolist()
joblib.dump(feature_cols, "xgb_feature_columns.joblib")
# holdout split (final evaluation)
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.20, random_state=RANDOM_STATE
)
print("Train:", X_train.shape, "Test:", X_test.shape)
# -------------------------
# Optuna objective (minimize CV MAE)
# -------------------------
def objective(trial):
params = {
"verbosity": 0,
"random_state": RANDOM_STATE,
"objective": "reg:squarederror",
# tree params
"max_depth": trial.suggest_int("max_depth", 3, 12),
"min_child_weight": trial.suggest_float("min_child_weight", 1e-1, 10.0, l
# sampling
"subsample": trial.suggest_float("subsample", 0.5, 1.0),
"colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
# regularization
"reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
"reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
# learning rate & trees
"learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.2),
# we'll set n_estimators at fit time using a fixed large value and rely o
"n_estimators": 1000,
# tree builder
"tree_method": "hist" # if your xgboost is older and doesn't support thi
}
model = xgb.XGBRegressor(**params)
# 5-fold CV on the training set
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
# use negative MAE because cross_val_score maximizes score; make_scorer retur
scores = -cross_val_score(model, X_train, y_train, cv=kf,
scoring=make_scorer(mean_absolute_error, greater_is
n_jobs=1)
return scores.mean() # minimize mean MAE
# -------------------------
# Run Optuna study
# -------------------------
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESamp
print("Starting Optuna tuning (this may take some minutes)...")
study.optimize(objective, n_trials=60, show_progress_bar=True)
print("Best trial:")
print(study.best_trial.params)
best_params = study.best_trial.params
# Some cleanup for final model: set practical n_estimators and other stable param
final_params = {
"random_state": RANDOM_STATE,
"objective": "reg:squarederror",
"tree_method": "hist",
"n_estimators": 1200, # slightly larger, gives room for learning_rate
**best_params
}
# Because Optuna returned learning_rate, max_depth etc., final_params contains th
# If older xgboost complains about tree_method or loguniform types, adjust accord
# -------------------------
# Train final model with best params
# -------------------------
print("\nTraining final XGBoost with best params...")
xgb_final = xgb.XGBRegressor(**final_params)
xgb_final.fit(X_train, y_train) # no eval_set for maximum compat
# Predictions
y_pred = xgb_final.predict(X_test)
y_pred_train = xgb_final.predict(X_train)
# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_r2 = r2_score(y_train, y_pred_train)
print("\n--- Final XGBoost Evaluation (test set) ---")
print(f"MAE : {mae:.6f} km")
print(f"RMSE: {rmse:.6f} km")
print(f"R² : {r2:.6f}")
print("\n--- Final XGBoost Evaluation (train set) ---")
print(f"Train MAE : {train_mae:.6f} km")
print(f"Train RMSE: {train_rmse:.6f} km")
print(f"Train R² : {train_r2:.6f}")
# Save model
joblib.dump(xgb_final, "xgb_clearance_model_tuned.joblib")

joblib.dump(
    {
        "study": study.best_trial.params,
        "results": {
            "test_mae": mae,
            "test_rmse": rmse,
            "test_r2": r2,
        },
    },
    "xgb_tuning_results.joblib"
)

print("\nSaved tuned model and tuning results.")
