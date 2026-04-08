"""
=============================================================
Evaluation Metrics — RMSE · MAE · MAPE
=============================================================
"""
import numpy as np
import pandas as pd
from typing import Dict


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def mape(y_true, y_pred, epsilon=1e-8) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = np.abs(y_true) > epsilon
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae" : round(mae(y_true, y_pred),  4),
        "mape": round(mape(y_true, y_pred), 4),
    }


def build_comparison_table(results: dict) -> pd.DataFrame:
    """Build a formatted model comparison table."""
    rows = []
    for name, r in results.items():
        row = {"Model": name}
        row.update(r.get("metrics", {}))
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    df = df.sort_values("rmse")
    df.columns = ["RMSE", "MAE", "MAPE (%)"]
    return df.round(3)
