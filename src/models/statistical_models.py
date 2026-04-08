"""
=============================================================
 Phase 3 — Statistical Time-Series Models
 Models: ARIMA · SARIMA · SARIMAX
 Cloud mapping: SageMaker Training Jobs / Vertex AI Training
=============================================================
"""

import numpy as np
import pandas as pd
import warnings, logging, sys
from pathlib import Path
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import MODEL_CONFIG
from src.evaluation.metrics import compute_metrics

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Stationarity check
# ─────────────────────────────────────────────────────────────
def check_stationarity(series: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller test.
    H0: series has a unit root (non-stationary)
    Reject H0 (p < 0.05) → stationary → d=0
    Fail to reject → difference once → d=1
    """
    result = adfuller(series.dropna(), autolag="AIC")
    out = {
        "adf_stat"  : round(result[0], 4),
        "p_value"   : round(result[1], 4),
        "n_lags"    : result[2],
        "stationary": result[1] < 0.05,
    }
    log.info(f"ADF stat={out['adf_stat']} p={out['p_value']} stationary={out['stationary']}")
    return out


def suggest_differencing(series: pd.Series) -> int:
    """Return d=0 if stationary, d=1 after first diff."""
    if check_stationarity(series)["stationary"]:
        return 0
    if check_stationarity(series.diff().dropna())["stationary"]:
        return 1
    return 2


# ─────────────────────────────────────────────────────────────
# ARIMA grid search
# ─────────────────────────────────────────────────────────────
def fit_arima(train: pd.Series, val: pd.Series,
              p_range=(0,3), q_range=(0,3)) -> dict:
    """
    Auto ARIMA via AIC-based grid search over (p, d, q).
    d is determined automatically via ADF test.
    """
    d = suggest_differencing(train)
    best_aic, best_order, best_model = np.inf, None, None

    for p, q in product(range(*p_range), range(*q_range)):
        try:
            m = ARIMA(train, order=(p, d, q)).fit()
            if m.aic < best_aic:
                best_aic, best_order, best_model = m.aic, (p, d, q), m
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("ARIMA grid search failed — no model converged.")

    forecast = best_model.forecast(steps=len(val))
    metrics  = compute_metrics(val.values, forecast.values)
    log.info(f"ARIMA best order={best_order} AIC={best_aic:.1f} | {metrics}")
    return {"model": best_model, "order": best_order, "aic": best_aic,
            "forecast": forecast, "metrics": metrics, "name": "ARIMA"}


# ─────────────────────────────────────────────────────────────
# SARIMA grid search
# ─────────────────────────────────────────────────────────────
def fit_sarima(train: pd.Series, val: pd.Series,
               seasonal_period: int = 7) -> dict:
    """
    SARIMA = ARIMA + seasonal component (P, D, Q)s.
    Retail has weekly seasonality → period=7.
    Searches a reduced grid to stay tractable.
    """
    d  = suggest_differencing(train)
    D  = 1   # one seasonal difference
    best_aic, best_order, best_seasonal, best_model = np.inf, None, None, None

    for p, q, P, Q in product([0,1,2], [0,1,2], [0,1], [0,1]):
        try:
            m = SARIMAX(
                train,
                order=(p, d, q),
                seasonal_order=(P, D, Q, seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            if m.aic < best_aic:
                best_aic   = m.aic
                best_order    = (p, d, q)
                best_seasonal = (P, D, Q, seasonal_period)
                best_model    = m
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("SARIMA grid search failed.")

    forecast = best_model.forecast(steps=len(val))
    metrics  = compute_metrics(val.values, forecast.values)
    log.info(f"SARIMA best={best_order}x{best_seasonal} AIC={best_aic:.1f} | {metrics}")
    return {"model": best_model, "order": best_order,
            "seasonal_order": best_seasonal, "aic": best_aic,
            "forecast": forecast, "metrics": metrics, "name": "SARIMA"}


# ─────────────────────────────────────────────────────────────
# SARIMAX — with exogenous variables
# ─────────────────────────────────────────────────────────────
def fit_sarimax(train: pd.Series, val: pd.Series,
                train_exog: pd.DataFrame, val_exog: pd.DataFrame,
                seasonal_period: int = 7) -> dict:
    """
    SARIMAX adds exogenous regressors (promotion, price, holidays)
    to the SARIMA model. This is the most powerful statistical model.
    Exog columns: promotion, price, is_weekend
    """
    d  = suggest_differencing(train)
    # Use best SARIMA structure as starting point (simplified for speed)
    order          = (1, d, 1)
    seasonal_order = (1, 1, 1, seasonal_period)

    try:
        m = SARIMAX(
            train,
            exog=train_exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    except Exception as e:
        log.warning(f"SARIMAX failed: {e}. Falling back to SARIMA.")
        return fit_sarima(train, val, seasonal_period)

    forecast = m.forecast(steps=len(val), exog=val_exog)
    metrics  = compute_metrics(val.values, forecast.values)
    log.info(f"SARIMAX order={order}x{seasonal_order} AIC={m.aic:.1f} | {metrics}")
    return {"model": m, "order": order, "seasonal_order": seasonal_order,
            "aic": m.aic, "forecast": forecast, "metrics": metrics, "name": "SARIMAX"}


# ─────────────────────────────────────────────────────────────
# Run all statistical models on one store-SKU series
# ─────────────────────────────────────────────────────────────
def run_statistical_models(df: pd.DataFrame,
                            store_id: int = 1,
                            sku_id: int   = 1,
                            test_date: str = "2023-07-01",
                            val_date:  str = "2023-01-01") -> dict:
    """
    Statistical models operate on a single time series.
    In production, this is parallelised across all store-SKU pairs
    using SageMaker Processing / Spark UDFs.
    """
    series = (
        df[(df["store_id"] == store_id) & (df["sku_id"] == sku_id)]
        .sort_values("date")
        .set_index("date")["demand"]
        .interpolate()
    )

    train = series[series.index <  val_date]
    val   = series[(series.index >= val_date) & (series.index < test_date)]

    # Exogenous for SARIMAX
    exog_cols = ["promotion", "is_weekend"]
    df_s = df[(df["store_id"] == store_id) & (df["sku_id"] == sku_id)].copy()
    df_s["is_weekend"] = pd.to_datetime(df_s["date"]).dt.dayofweek >= 5
    df_s = df_s.set_index("date").sort_index()
    train_exog = df_s.loc[df_s.index <  val_date, exog_cols]
    val_exog   = df_s.loc[(df_s.index >= val_date) & (df_s.index < test_date), exog_cols]

    results = {}
    log.info("Fitting ARIMA...")
    results["ARIMA"]   = fit_arima(train, val)
    log.info("Fitting SARIMA...")
    results["SARIMA"]  = fit_sarima(train, val)
    log.info("Fitting SARIMAX...")
    results["SARIMAX"] = fit_sarimax(train, val, train_exog, val_exog)

    return results, series, val


if __name__ == "__main__":
    from src.data.data_generator import generate_retail_dataset
    df = generate_retail_dataset(save=False)
    results, series, val = run_statistical_models(df)
    for name, r in results.items():
        print(f"{name}: RMSE={r['metrics']['rmse']:.2f} MAE={r['metrics']['mae']:.2f} MAPE={r['metrics']['mape']:.2f}%")
