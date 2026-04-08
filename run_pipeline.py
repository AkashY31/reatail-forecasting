"""
=============================================================
 MASTER PIPELINE — Retail Demand Forecasting
 Runs all 7 phases end-to-end in correct order.

 Usage:
   python run_pipeline.py                  # full run
   python run_pipeline.py --phase 1        # single phase
   python run_pipeline.py --store 1 --sku 1

 Cloud: This script = your SageMaker Pipeline definition /
        Azure ML Pipeline / Vertex AI Pipeline YAML
=============================================================
"""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.config import (DATA_DIR, OUTPUT_DIR, MODEL_CONFIG,
                              DRIFT_CONFIG)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log", mode="a"),
    ]
)
log = logging.getLogger(__name__)


def banner(msg: str):
    log.info("=" * 60)
    log.info(f"  {msg}")
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────
# Phase 1: Data generation + EDA
# ─────────────────────────────────────────────────────────────
def run_phase1() -> pd.DataFrame:
    banner("PHASE 1 — Data Generation + EDA")
    from src.data.data_generator import generate_retail_dataset
    from src.data.eda import run_full_eda

    df = generate_retail_dataset()
    run_full_eda(df, save_plots=True)
    return df


# ─────────────────────────────────────────────────────────────
# Phase 2: Preprocessing + feature engineering
# ─────────────────────────────────────────────────────────────
def run_phase2(df: pd.DataFrame):
    banner("PHASE 2 — Preprocessing + Feature Engineering")
    from src.features.preprocessing import (
        build_preprocessing_pipeline, get_feature_columns,
        temporal_train_test_split
    )
    pipe = build_preprocessing_pipeline()
    df_proc = pipe.fit_transform(df)

    feat_cols = get_feature_columns(df_proc)
    log.info(f"Feature count: {len(feat_cols)}")

    splits = temporal_train_test_split(
        df_proc,
        test_date=MODEL_CONFIG["test_split_date"],
        val_date =MODEL_CONFIG["val_split_date"],
        feature_cols=feat_cols,
    )
    df_proc.to_parquet(DATA_DIR / "processed" / "features.parquet", index=False)
    return df_proc, feat_cols, splits, pipe


# ─────────────────────────────────────────────────────────────
# Phase 3: Statistical models (ARIMA / SARIMA / SARIMAX)
# ─────────────────────────────────────────────────────────────
def run_phase3(df: pd.DataFrame, store_id: int = 1, sku_id: int = 1):
    banner("PHASE 3 — Statistical Models (ARIMA / SARIMA / SARIMAX)")
    from src.models.statistical_models import run_statistical_models
    results, series, val = run_statistical_models(
        df, store_id=store_id, sku_id=sku_id,
        test_date=MODEL_CONFIG["test_split_date"],
        val_date =MODEL_CONFIG["val_split_date"],
    )
    for name, r in results.items():
        log.info(f"  {name}: RMSE={r['metrics']['rmse']:.2f} "
                  f"MAE={r['metrics']['mae']:.2f} "
                  f"MAPE={r['metrics']['mape']:.2f}%")
    return results, series, val


# ─────────────────────────────────────────────────────────────
# Phase 4: ML / DL models
# ─────────────────────────────────────────────────────────────
def run_phase4(splits, df_proc: pd.DataFrame, n_trials: int = 10):
    banner("PHASE 4 — ML Models (LR / RF / XGBoost / LSTM)")
    from src.models.ml_models import run_ml_models, save_models

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = splits

    # Use a single store-SKU series for LSTM
    series = (
        df_proc[(df_proc["store_id"] == 1) & (df_proc["sku_id"] == 1)]
        .sort_values("date")["demand"].values
    )
    val_start = int(len(series) * 0.7)
    test_start = int(len(series) * 0.85)
    train_series = series[:val_start]
    val_series   = series[val_start:test_start]

    results = run_ml_models(
        X_tr, y_tr, X_va, y_va,
        train_series=train_series,
        val_series=val_series,
        n_trials=n_trials,
    )

    model_path = OUTPUT_DIR / "models"
    save_models(results, model_path)

    for name, r in results.items():
        log.info(f"  {name}: RMSE={r['metrics']['rmse']:.2f} "
                  f"MAE={r['metrics']['mae']:.2f} "
                  f"MAPE={r['metrics']['mape']:.2f}%")

    return results, (X_va, y_va)


# ─────────────────────────────────────────────────────────────
# Phase 5: Ensemble + evaluation
# ─────────────────────────────────────────────────────────────
def run_phase5(ml_results: dict, stat_results: dict,
               y_val: np.ndarray, df_proc: pd.DataFrame):
    banner("PHASE 5 — Ensemble + Full Evaluation")
    from src.evaluation.ensemble import (
        weighted_ensemble, stacking_ensemble, run_evaluation
    )

    # Combine all models into one results dict for ensemble
    all_preds = {**ml_results}
    # Add statistical model results (already have preds)
    for name, r in stat_results.items():
        if r.get("forecast") is not None:
            r["preds"] = r["forecast"].values
            all_preds[name] = r

    w_ensemble = weighted_ensemble(all_preds, y_val)
    s_ensemble = stacking_ensemble(all_preds, y_val)

    all_results = {**all_preds,
                   "Weighted Ensemble": w_ensemble,
                   "Stacking Ensemble": s_ensemble}

    val_dates = pd.to_datetime(
        df_proc[df_proc["date"] >= MODEL_CONFIG["val_split_date"]]["date"].unique()
    )

    eval_out = run_evaluation(
        all_results, y_val, val_dates=val_dates,
        rf_result=ml_results.get("Random Forest"),
        xgb_result=ml_results.get("XGBoost"),
        save=True,
    )

    log.info(f"\nBest model: {eval_out['best']}")
    log.info("\n" + eval_out["table"].to_string())
    return all_results, eval_out


# ─────────────────────────────────────────────────────────────
# Phase 6: Drift detection
# ─────────────────────────────────────────────────────────────
def run_phase6(df: pd.DataFrame):
    banner("PHASE 6 — Drift Detection + Monitoring")
    from src.monitoring.drift_detection import (
        simulate_data_drift, run_drift_monitoring
    )

    # Simulate gradual drift after Oct 2023
    df_drifted = simulate_data_drift(df, drift_start="2023-10-01",
                                      drift_type="gradual")

    ref_window = DRIFT_CONFIG["reference_window"]
    det_window = DRIFT_CONFIG["detection_window"]

    df_ref = df[df["date"] < "2023-07-01"].tail(ref_window * 500)
    df_cur = df_drifted[df_drifted["date"] >= "2023-10-01"].head(det_window * 500)

    report = run_drift_monitoring(df_ref, df_cur, save=True)
    log.info(f"Drift PSI={report['psi_score']} ({report['psi_status']})")
    log.info(f"KS test: {report['ks_result']}")
    return report


# ─────────────────────────────────────────────────────────────
# Phase 7: Dashboard info
# ─────────────────────────────────────────────────────────────
def run_phase7():
    banner("PHASE 7 — Streamlit Dashboard")
    log.info("Dashboard is ready.")
    log.info("To launch locally:")
    log.info("  streamlit run src/serving/dashboard.py")
    log.info("To deploy on cloud:")
    log.info("  docker build -t retail-forecast .")
    log.info("  docker push <your-registry>/retail-forecast")
    log.info("  # Then deploy to ECS / AKS / Cloud Run")


# ─────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────
def main(args):
    Path("logs").mkdir(exist_ok=True)
    t0 = time.time()
    banner("RETAIL DEMAND FORECASTING — FULL PIPELINE START")

    phase = args.phase  # None = run all

    df      = None
    splits  = None
    df_proc = None
    feat_cols = None

    if phase in (None, 1):
        df = run_phase1()

    if phase in (None, 2):
        if df is None:
            df = pd.read_parquet(DATA_DIR / "raw" / "retail_sales.parquet")
        df_proc, feat_cols, splits, pipe = run_phase2(df)

    if phase in (None, 3):
        if df is None:
            df = pd.read_parquet(DATA_DIR / "raw" / "retail_sales.parquet")
        stat_results, series, val = run_phase3(df, store_id=args.store,
                                                sku_id=args.sku)
    else:
        stat_results = {}

    if phase in (None, 4):
        if splits is None:
            log.error("Run phase 2 first to generate splits.")
            return
        ml_results, (X_va, y_va) = run_phase4(
            splits, df_proc, n_trials=args.n_trials
        )
    else:
        ml_results = {}
        y_va = np.array([])

    if phase in (None, 5):
        if not ml_results:
            log.warning("No ML results — skipping ensemble.")
        else:
            all_results, eval_out = run_phase5(
                ml_results, stat_results, y_va.values, df_proc
            )

    if phase in (None, 6):
        if df is None:
            df = pd.read_parquet(DATA_DIR / "raw" / "retail_sales.parquet")
        run_phase6(df)

    if phase in (None, 7):
        run_phase7()

    elapsed = round(time.time() - t0, 1)
    banner(f"PIPELINE COMPLETE in {elapsed}s")
    log.info(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",    type=int, default=None,
                        help="Run a specific phase (1-7). Default: all")
    parser.add_argument("--store",    type=int, default=1)
    parser.add_argument("--sku",      type=int, default=1)
    parser.add_argument("--n_trials", type=int, default=10,
                        help="Optuna HPT trials for ML models")
    args = parser.parse_args()
    main(args)
