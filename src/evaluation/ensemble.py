"""
=============================================================
 Phase 5 — Ensemble Methods + Model Evaluation
 Ensemble: Weighted Average · Stacking (meta-learner)
 Cloud   : Results logged to MLflow on SageMaker / Azure ML
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging, sys, mlflow, mlflow.sklearn
from pathlib import Path
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import OUTPUT_DIR, MLFLOW_URI, MLFLOW_EXPERIMENT
from src.evaluation.metrics import compute_metrics, build_comparison_table

log = logging.getLogger(__name__)
PALETTE = ["#185FA5","#0F6E56","#854F0B","#993C1D","#534AB7","#1D9E75"]


# ─────────────────────────────────────────────────────────────
# 1. Weighted average ensemble
# ─────────────────────────────────────────────────────────────
def weighted_ensemble(predictions: dict, y_true: np.ndarray,
                       method: str = "rmse_inverse") -> dict:
    """
    Combine predictions by weighting each model inversely
    proportional to its validation RMSE.
    Better models get higher weights automatically.
    """
    preds  = {k: np.array(v["preds"]) for k, v in predictions.items()
               if v.get("preds") is not None}
    rmses  = {k: v["metrics"]["rmse"] for k, v in predictions.items()
               if v.get("preds") is not None}

    # Align lengths (LSTM may predict fewer steps due to window)
    min_len = min(len(p) for p in preds.values())
    preds   = {k: v[-min_len:] for k, v in preds.items()}
    y_eval  = y_true[-min_len:]

    inv_rmse = {k: 1.0 / (v + 1e-8) for k, v in rmses.items()}
    total    = sum(inv_rmse.values())
    weights  = {k: v / total for k, v in inv_rmse.items()}

    ensemble_pred = sum(w * preds[k] for k, w in weights.items())
    ensemble_pred = ensemble_pred.clip(0)
    metrics = compute_metrics(y_eval, ensemble_pred)

    log.info(f"Weighted Ensemble weights={weights}")
    log.info(f"Weighted Ensemble metrics={metrics}")

    return {"preds": ensemble_pred, "weights": weights,
            "metrics": metrics, "y_eval": y_eval,
            "name": "Weighted Ensemble"}


# ─────────────────────────────────────────────────────────────
# 2. Stacking ensemble (meta-learner)
# ─────────────────────────────────────────────────────────────
def stacking_ensemble(predictions: dict, y_true: np.ndarray) -> dict:
    """
    Stacking: treat base model predictions as features,
    train a Ridge meta-learner to combine them optimally.
    Learns non-linear combinations — usually beats simple averaging.
    """
    valid_preds = {k: np.array(v["preds"]) for k, v in predictions.items()
                   if v.get("preds") is not None}
    min_len = min(len(p) for p in valid_preds.values())
    aligned = {k: v[-min_len:] for k, v in valid_preds.items()}
    y_eval  = y_true[-min_len:]

    # Stack as feature matrix
    X_stack = np.column_stack(list(aligned.values()))

    # Train meta-learner on first 70% of validation, test on rest
    split   = int(0.7 * len(y_eval))
    meta    = Ridge(alpha=1.0, positive=True)   # positive=True → no negative weights
    meta.fit(X_stack[:split], y_eval[:split])

    stacked_pred = meta.predict(X_stack[split:]).clip(0)
    metrics = compute_metrics(y_eval[split:], stacked_pred)

    coef_map = dict(zip(aligned.keys(), meta.coef_))
    log.info(f"Stacking meta-learner coefs={coef_map}")
    log.info(f"Stacking metrics={metrics}")

    return {"model": meta, "preds": stacked_pred,
            "coef": coef_map, "metrics": metrics, "name": "Stacking Ensemble"}


# ─────────────────────────────────────────────────────────────
# 3. MLflow experiment logging
# ─────────────────────────────────────────────────────────────
def log_to_mlflow(all_results: dict):
    """
    Log every model's metrics and params to MLflow.
    Cloud: set MLFLOW_URI to your SageMaker / Azure ML / Vertex tracking server.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    for name, r in all_results.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_metrics(r.get("metrics", {}))
            if r.get("best_params"):
                mlflow.log_params(r["best_params"])
            mlflow.set_tag("model_type", name)
    log.info("All runs logged to MLflow.")


# ─────────────────────────────────────────────────────────────
# 4. Visualizations
# ─────────────────────────────────────────────────────────────
def plot_model_comparison(all_results: dict, y_val: np.ndarray,
                           val_dates: pd.DatetimeIndex) -> plt.Figure:
    """
    4-panel comparison:
      - Actual vs predicted (all models)
      - RMSE bar chart
      - MAE bar chart
      - MAPE bar chart
    """
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle("Model Comparison — Retail Demand Forecasting",
                 fontsize=15, fontweight="bold")

    # Panel 1: Forecast overlay
    ax1 = fig.add_subplot(gs[0, :])
    min_len = min(
        len(r["preds"]) for r in all_results.values() if r.get("preds") is not None
    )
    dates_plot = val_dates[-min_len:] if val_dates is not None else range(min_len)
    ax1.plot(dates_plot, y_val[-min_len:], color="black", linewidth=1.8,
             label="Actual", zorder=5)
    for i, (name, r) in enumerate(all_results.items()):
        if r.get("preds") is None:
            continue
        preds = r["preds"][-min_len:]
        ax1.plot(dates_plot, preds, color=PALETTE[i % len(PALETTE)],
                 linewidth=1.2, alpha=0.75, label=name)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.set_title("Actual vs Forecasted demand", fontsize=12)
    ax1.set_ylabel("Units", fontsize=10)

    # Panels 2–4: Metric bar charts
    metrics_names = ["rmse", "mae", "mape"]
    metric_labels = ["RMSE", "MAE", "MAPE (%)"]
    for idx, (mname, mlabel) in enumerate(zip(metrics_names, metric_labels)):
        ax = fig.add_subplot(gs[1, idx % 2]) if idx < 2 else fig.add_subplot(gs[1, 1])
        names  = list(all_results.keys())
        values = [all_results[n]["metrics"].get(mname, 0) for n in names]
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
        bars   = ax.bar(names, values, color=colors, alpha=0.8, edgecolor="white")
        ax.set_title(f"{mlabel} by model", fontsize=11)
        ax.set_ylabel(mlabel, fontsize=9)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.01,
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=8)

    return fig


def plot_feature_importance(rf_result: dict, xgb_result: dict) -> plt.Figure:
    """Side-by-side feature importance for RF and XGBoost."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Feature Importance", fontsize=14, fontweight="bold")

    for ax, result, color in zip(axes,
                                  [rf_result, xgb_result],
                                  [PALETTE[0], PALETTE[2]]):
        fi = result.get("feature_importance")
        if fi is None or len(fi) == 0:
            ax.set_title(f"{result['name']} — no importance available")
            continue
        top15 = fi.head(15)
        ax.barh(top15.index[::-1], top15.values[::-1],
                color=color, alpha=0.8)
        ax.set_title(f"{result['name']} — top 15 features", fontsize=11)
        ax.set_xlabel("Importance score", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    return fig


def plot_residuals(all_results: dict, y_val: np.ndarray) -> plt.Figure:
    """Residual distribution per model."""
    n   = len([r for r in all_results.values() if r.get("preds") is not None])
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    fig.suptitle("Residual Analysis", fontsize=14, fontweight="bold")
    if n == 1:
        axes = [axes]
    ax_iter = iter(axes)
    for name, r in all_results.items():
        if r.get("preds") is None:
            continue
        ax   = next(ax_iter)
        preds = np.array(r["preds"])
        y_ev  = y_val[-len(preds):]
        resid = y_ev - preds
        ax.hist(resid, bins=40, color=PALETTE[list(all_results).index(name) % len(PALETTE)],
                alpha=0.75, edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
        ax.set_title(f"{name}\nμ={resid.mean():.1f} σ={resid.std():.1f}", fontsize=10)
        ax.set_xlabel("Residual", fontsize=9)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# Master evaluation runner
# ─────────────────────────────────────────────────────────────
def run_evaluation(all_results: dict, y_val: np.ndarray,
                   val_dates: pd.DatetimeIndex = None,
                   rf_result: dict = None, xgb_result: dict = None,
                   save: bool = True) -> dict:
    out_dir = OUTPUT_DIR / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = build_comparison_table(all_results)
    log.info("\n" + "=" * 50)
    log.info("MODEL COMPARISON TABLE")
    log.info("=" * 50)
    log.info("\n" + table.to_string())
    table.to_csv(out_dir / "model_comparison.csv")

    figs = {}
    figs["comparison"] = plot_model_comparison(all_results, y_val, val_dates)
    if rf_result and xgb_result:
        figs["feature_importance"] = plot_feature_importance(rf_result, xgb_result)
    figs["residuals"] = plot_residuals(all_results, y_val)

    if save:
        for name, fig in figs.items():
            path = out_dir / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info(f"Saved → {path}")

    try:
        log_to_mlflow(all_results)
    except Exception as e:
        log.warning(f"MLflow logging skipped: {e}")

    best_model_name = table.index[0]
    log.info(f"\nBest model: {best_model_name} "
              f"(RMSE={table.loc[best_model_name,'RMSE']:.3f})")

    return {"table": table, "best": best_model_name, "figures": figs}
