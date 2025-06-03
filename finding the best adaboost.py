"""
Daily Listening Time Prediction – Optimized AdaBoost Pipeline
============================================================
This script keeps **only** the AdaBoostRegressor, performs a randomized
hyper‑parameter search to find the best configuration, then trains the
final model, evaluates it, draws scatter + residual plots, and saves all
artifacts (plots, metrics, and the trained model) to disk.
"""
from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from scipy.stats import randint, uniform
from joblib import parallel_backend




# Configuration                                  
DATA_PATH = "D:\\Desktop\\机器学习\\project\\Global_Music_Streaming_Listener_Preferences.csv"  
TARGET = "Minutes Streamed Per Day" 
RANDOM_STATE = 42
N_ITER_SEARCH = 40  # hyper‑parameter search iterations
CV_FOLDS = 5
TEST_SIZE = 0.2

                                     
# 1. Load data                                    
print("📥 Loading data…")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

                                   
# 2. Feature/target split & log‑transform target                                     
print("🔄 Preparing features & target…")
y = np.log1p(df[TARGET])  # log transform stabilises variance
X = df.drop(columns=[TARGET])

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ],
    remainder="drop",
)
                                   
# 3. Build AdaBoost pipeline                                     -
print("🛠️  Building AdaBoost pipeline…")
base_tree = DecisionTreeRegressor(
    random_state=RANDOM_STATE, max_depth=3, min_samples_leaf=5
)

ada = AdaBoostRegressor(estimator=base_tree, random_state=RANDOM_STATE)

pipe = Pipeline([
    ("pre", preprocess),
    ("model", ada),
])

param_distributions = {
    "model__n_estimators": randint(50, 400),
    "model__learning_rate": uniform(0.01, 1.0),
    "model__loss": ["linear", "square", "exponential"],
    "model__estimator__max_depth": [2, 3, 4, 5, None],
    "model__estimator__min_samples_leaf": randint(1, 20),
}

                                    
# 4. Train/validation split & hyper‑parameter search
print("🔍 Starting randomized search…")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=N_ITER_SEARCH,
    cv=CV_FOLDS,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=3,
)

search.fit(X_train, y_train)
print("✅ Search complete.")
print("Best params:", search.best_params_)

best_model = search.best_estimator_


# 5. Evaluation helpers

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, R‑squared as a dict."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "R2": r2_score(y_true, y_pred),
    }

def evaluate_and_plot(model, X_tr, y_tr, X_te, y_te):
    """Evaluate model and save scatter+residual plot."""
    preds_tr = model.predict(X_tr)
    preds_te = model.predict(X_te)

    met_tr = _metrics(y_tr, preds_tr)
    met_te = _metrics(y_te, preds_te)

    metrics_df = pd.DataFrame([
        {"set": "train", **met_tr},
        {"set": "test", **met_te},
    ])
    print("\n📊 Metrics:\n", metrics_df.to_string(index=False))

    #  Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Scatter plot
    ax[0].scatter(y_te, preds_te, alpha=0.6)
    lims = [min(y_te.min(), preds_te.min()), max(y_te.max(), preds_te.max())]
    ax[0].plot(lims, lims, " ", linewidth=1)
    ax[0].set_xlabel("True log(1+Minutes)")
    ax[0].set_ylabel("Predicted")
    ax[0].set_title("Predicted vs True")

    # (b) Residuals
    residuals = y_te - preds_te
    ax[1].scatter(preds_te, residuals, alpha=0.6)
    ax[1].hlines(0, preds_te.min(), preds_te.max(), colors="black", linestyles="dashed")
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("Residual (True − Pred)")
    ax[1].set_title("Residuals")

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", "ada_scatter_residual.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()
    return metrics_df, plot_path



# 6. Run evaluation
print("🧮 Evaluating best model…")
metrics_df, plot_file = evaluate_and_plot(
    best_model, X_train, y_train, X_test, y_test
)


# 7. Save artefacts
print("💾 Saving artefacts…")
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/ada_best.joblib")
metrics_df.to_csv("models/ada_metrics.csv", index=False)

print(f"All done! Plot → {plot_file}, model & metrics → models/ directory.")
