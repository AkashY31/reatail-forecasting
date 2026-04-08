"""
=============================================================
Retail Demand Forecasting — Central Configuration
Cloud mapping: This file = AWS SSM Parameter Store /
                Azure App Config / GCP Secret Manager
=============================================================
"""

import os
from pathlib import Path

# ── Project root (works locally and in Docker/cloud) ────────
ROOT_DIR   = Path(__file__).resolve().parents[1]
DATA_DIR   = Path(os.getenv("DATA_PATH",   ROOT_DIR / "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_PATH", ROOT_DIR / "outputs"))
LOG_DIR    = Path(os.getenv("LOG_PATH",    ROOT_DIR / "logs"))

for d in [DATA_DIR / "raw", DATA_DIR / "processed", OUTPUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── MLflow (local default → override with cloud URI) ────────
MLFLOW_URI        = os.getenv("MLFLOW_URI", str(ROOT_DIR / "mlruns"))
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "retail_demand_forecasting")

# ── Data generation parameters ───────────────────────────────
DATA_CONFIG = {
    "n_stores"       : 10,
    "n_skus"         : 50,
    "start_date"     : "2020-01-01",
    "end_date"       : "2023-12-31",
    "random_seed"    : 42,
    "base_demand"    : 100,
}

# ── Feature engineering ───────────────────────────────────────
FEATURE_CONFIG = {
    "lag_days"       : [1, 7, 14, 28, 56],
    "rolling_windows": [7, 14, 28],
    "fourier_terms"  : 3,          # sin/cos pairs for seasonality
    "target_col"     : "demand",
    "date_col"       : "date",
}

# ── Model parameters ──────────────────────────────────────────
MODEL_CONFIG = {
    "test_split_date"  : "2023-07-01",   # temporal split — no leakage
    "val_split_date"   : "2023-01-01",
    "forecast_horizon" : 28,             # days ahead to forecast
    "cv_splits"        : 5,
}

# ── Evaluation ────────────────────────────────────────────────
EVAL_METRICS = ["rmse", "mae", "mape"]

# ── Drift detection ───────────────────────────────────────────
DRIFT_CONFIG = {
    "psi_threshold"    : 0.2,   # Population Stability Index
    "ks_alpha"         : 0.05,  # Kolmogorov-Smirnov p-value threshold
    "reference_window" : 90,    # days of reference data
    "detection_window" : 30,    # days of new data to test
}
