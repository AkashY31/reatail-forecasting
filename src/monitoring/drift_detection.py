"""
=============================================================
 Phase 6 — Data Drift Detection & Production Monitoring
 Methods: PSI (Population Stability Index) · KS Test · Z-Score
 Cloud  : SageMaker Model Monitor / Azure ML Data Drift /
           Vertex AI Model Monitoring
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import logging, sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import DRIFT_CONFIG, OUTPUT_DIR

log = logging.getLogger(__name__)
PALETTE = ["#185FA5","#0F6E56","#854F0B","#993C1D","#534AB7"]


# ─────────────────────────────────────────────────────────────
# Drift simulation
# ─────────────────────────────────────────────────────────────
def simulate_data_drift(df: pd.DataFrame, drift_start: str = "2023-10-01",
                         drift_type: str = "gradual") -> pd.DataFrame:
    """
    Simulate 3 real-world drift scenarios:
      sudden  — demand jumps sharply (e.g. viral product)
      gradual — slow demand erosion (e.g. market saturation)
      seasonal_shift — seasonality pattern changes (e.g. new holiday)
    """
    df = df.copy()
    drift_mask = df["date"] >= drift_start

    if drift_type == "sudden":
        # 40% demand spike — sudden viral event
        df.loc[drift_mask, "demand"] *= 1.4
        log.info(f"Applied SUDDEN drift from {drift_start}: +40% demand")

    elif drift_type == "gradual":
        # Linear demand decline over drift period
        n_days = drift_mask.sum()
        decay  = np.linspace(1.0, 0.6, n_days)
        df.loc[drift_mask, "demand"] *= np.tile(
            decay, (df["store_id"].nunique() * df["sku_id"].nunique(), 1)
        ).flatten()[:drift_mask.sum()]
        log.info(f"Applied GRADUAL drift from {drift_start}: demand decays to 60%")

    elif drift_type == "seasonal_shift":
        # Weekend effect disappears (behavioral change)
        mask = drift_mask & (pd.to_datetime(df["date"]).dt.dayofweek >= 5)
        df.loc[mask, "demand"] *= 0.7
        log.info(f"Applied SEASONAL SHIFT drift: weekend demand -30%")

    return df


# ─────────────────────────────────────────────────────────────
# PSI — Population Stability Index
# ─────────────────────────────────────────────────────────────
def compute_psi(reference: np.ndarray, current: np.ndarray,
                n_bins: int = 10) -> Tuple[float, pd.DataFrame]:
    """
    PSI measures how much a distribution has shifted.
    Rule of thumb:
      PSI < 0.1  → No significant change
      PSI < 0.2  → Moderate change, monitor
      PSI >= 0.2 → Significant drift → retrain
    """
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]

    bins = np.percentile(ref_clean, np.linspace(0, 100, n_bins + 1))
    bins[0]  -= 1e-8
    bins[-1] += 1e-8

    ref_counts = np.histogram(ref_clean, bins=bins)[0]
    cur_counts = np.histogram(cur_clean, bins=bins)[0]

    # Avoid division by zero
    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / len(ref_clean))
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / len(cur_clean))

    psi_vals = (ref_pct - cur_pct) * np.log(ref_pct / cur_pct)
    psi_total = psi_vals.sum()

    detail = pd.DataFrame({
        "bin"        : range(n_bins),
        "ref_pct"    : ref_pct.round(4),
        "cur_pct"    : cur_pct.round(4),
        "psi_contrib": psi_vals.round(4),
    })
    return float(psi_total), detail


# ─────────────────────────────────────────────────────────────
# KS Test
# ─────────────────────────────────────────────────────────────
def compute_ks_test(reference: np.ndarray, current: np.ndarray) -> dict:
    """
    Kolmogorov-Smirnov 2-sample test.
    H0: reference and current come from the same distribution.
    p < 0.05 → reject H0 → drift detected.
    """
    stat, p_value = stats.ks_2samp(
        reference[~np.isnan(reference)],
        current[~np.isnan(current)]
    )
    drift = p_value < DRIFT_CONFIG["ks_alpha"]
    return {"ks_stat": round(stat, 4), "p_value": round(p_value, 4),
            "drift_detected": drift}


# ─────────────────────────────────────────────────────────────
# Z-Score based prediction drift
# ─────────────────────────────────────────────────────────────
def compute_prediction_drift(y_true_ref: np.ndarray, y_pred_ref: np.ndarray,
                              y_true_cur: np.ndarray, y_pred_cur: np.ndarray) -> dict:
    """
    Check if model performance has degraded significantly
    by comparing error distributions between reference and current windows.
    """
    err_ref = np.abs(y_true_ref - y_pred_ref)
    err_cur = np.abs(y_true_cur - y_pred_cur)
    z_stat, p_value = stats.ttest_ind(err_ref, err_cur, equal_var=False)
    return {
        "ref_mae"       : round(err_ref.mean(), 4),
        "cur_mae"       : round(err_cur.mean(), 4),
        "mae_change_pct": round((err_cur.mean() - err_ref.mean()) / (err_ref.mean() + 1e-8) * 100, 2),
        "z_stat"        : round(z_stat, 4),
        "p_value"       : round(p_value, 4),
        "drift_detected": p_value < 0.05,
    }


# ─────────────────────────────────────────────────────────────
# Full drift monitoring report
# ─────────────────────────────────────────────────────────────
def run_drift_monitoring(df_ref: pd.DataFrame, df_cur: pd.DataFrame,
                          save: bool = True) -> dict:
    """
    Run full drift monitoring pipeline on two data windows.
    Outputs: PSI score, KS test, distribution plots, alert flags.
    """
    ref_demand = df_ref["demand"].dropna().values
    cur_demand = df_cur["demand"].dropna().values

    psi_score, psi_detail = compute_psi(ref_demand, cur_demand)
    ks_result              = compute_ks_test(ref_demand, cur_demand)

    psi_flag = "ALERT" if psi_score >= DRIFT_CONFIG["psi_threshold"] else \
               "WARN"  if psi_score >= 0.1 else "OK"

    report = {
        "psi_score"     : round(psi_score, 4),
        "psi_status"    : psi_flag,
        "ks_result"     : ks_result,
        "psi_detail"    : psi_detail,
    }

    log.info(f"Drift Report — PSI={psi_score:.4f} ({psi_flag}) | "
              f"KS p={ks_result['p_value']:.4f} drift={ks_result['drift_detected']}")

    if save:
        fig = _plot_drift_report(ref_demand, cur_demand, report)
        out = OUTPUT_DIR / "monitoring"
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "drift_report.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Drift report saved → {out / 'drift_report.png'}")

    return report


def _plot_drift_report(ref: np.ndarray, cur: np.ndarray, report: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Data Drift Report — PSI={report['psi_score']:.4f} "
                  f"({report['psi_status']})",
                 fontsize=13, fontweight="bold")

    # Distribution overlay
    ax = axes[0]
    ax.hist(ref.clip(max=np.percentile(ref, 99)), bins=50,
            alpha=0.6, color=PALETTE[0], label="Reference", density=True)
    ax.hist(cur.clip(max=np.percentile(cur, 99)), bins=50,
            alpha=0.6, color=PALETTE[3], label="Current",   density=True)
    ax.legend(fontsize=9)
    ax.set_title("Demand distribution shift", fontsize=11)
    ax.set_xlabel("Demand", fontsize=9)

    # PSI by bin
    ax = axes[1]
    detail = report["psi_detail"]
    colors = ["#993C1D" if v >= 0.025 else "#185FA5" for v in detail["psi_contrib"]]
    ax.bar(detail["bin"], detail["psi_contrib"], color=colors, alpha=0.8)
    ax.axhline(0.025, color="orange", linestyle="--", linewidth=1.2, label="Bin alert")
    ax.legend(fontsize=9)
    ax.set_title("PSI contribution per bin", fontsize=11)
    ax.set_xlabel("Bin", fontsize=9)
    ax.set_ylabel("PSI contribution", fontsize=9)

    # CDF comparison (KS test visual)
    ax = axes[2]
    ref_sorted = np.sort(ref)
    cur_sorted = np.sort(cur)
    ax.plot(ref_sorted, np.linspace(0, 1, len(ref_sorted)),
            color=PALETTE[0], label="Reference", linewidth=1.5)
    ax.plot(cur_sorted, np.linspace(0, 1, len(cur_sorted)),
            color=PALETTE[3], label="Current",   linewidth=1.5)
    ax.legend(fontsize=9)
    ax.set_title(f"CDF comparison (KS p={report['ks_result']['p_value']:.3f})", fontsize=11)
    ax.set_xlabel("Demand", fontsize=9)
    ax.set_ylabel("Cumulative probability", fontsize=9)

    plt.tight_layout()
    return fig
